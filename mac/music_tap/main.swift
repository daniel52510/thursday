// THURSDAY Music Tap: YouTube Music Desktop's sound, streamed to thursday-core for the satellite.
//
// A Core Audio process tap (macOS 14.2+) on the app's own processes only: Chromium plays audio from
// a helper process, so every process whose binary lives in the app bundle is tapped. When core says
// mute (the music plays on the satellite), the tap mutes the app on the Mac too, so the soundbar or
// speakers stay quiet. Audio leaves as 24 kHz mono int16 in 40 ms binary WebSocket frames to
// ws://127.0.0.1:8000/ws/music-tap with the token in ~/.config/thursday/music_tap_token.
// Core sends {"mute": true|false}. Needs "System Audio Recording" permission (asked on first tap).
// Built and installed by scripts/build_music_tap.sh; run by launchd (com.thursday.musictap).

import AudioToolbox
import CoreAudio
import Darwin
import Foundation

let appMarker = "/YouTube Music Desktop App.app/"
let coreURL = ProcessInfo.processInfo.environment["MUSIC_TAP_URL"] ?? "ws://127.0.0.1:8000/ws/music-tap"
let outRate = 24000.0
let frameBytes = Int(outRate) * 2 * 40 / 1000  // 40 ms

func log(_ s: String) { FileHandle.standardError.write("music-tap: \(s)\n".data(using: .utf8)!) }

func readProp<T>(_ id: AudioObjectID, _ sel: AudioObjectPropertySelector, _ empty: T) -> [T] {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    var out = [T](repeating: empty, count: Int(size) / MemoryLayout<T>.stride)
    guard AudioObjectGetPropertyData(id, &addr, 0, nil, &size, &out) == noErr else { return [] }
    return out
}

/// The app's processes that Core Audio knows about (the ones that can make sound).
func appAudioProcesses() -> [AudioObjectID] {
    readProp(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList, AudioObjectID(0)).filter { obj in
        guard let pid = readProp(obj, kAudioProcessPropertyPID, pid_t(0)).first, pid > 0 else { return false }
        var buf = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        guard proc_pidpath(pid, &buf, UInt32(buf.count)) > 0 else { return false }
        return String(cString: buf).contains(appMarker)
    }
}

func defaultOutputUID() -> String? {
    guard let dev = readProp(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultSystemOutputDevice,
                             AudioObjectID(0)).first else { return nil }
    var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceUID, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var uid: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    guard AudioObjectGetPropertyData(dev, &addr, 0, nil, &size, &uid) == noErr, let u = uid else { return nil }
    return u.takeRetainedValue() as String
}

/// Float stereo at the tap's rate in; 24 kHz mono int16 out. A one-pole low-pass under the new
/// Nyquist, then linear interpolation: plenty for the satellite's speaker.
final class Converter {
    let ratio: Double
    var pos = 0.0, last: Float = 0, lp: Float = 0
    let alpha: Float
    init(inRate: Double) {
        ratio = inRate / outRate
        let rc = 1.0 / (2 * Double.pi * 10_000), dt = 1.0 / inRate
        alpha = Float(dt / (rc + dt))
    }
    func convert(_ ptr: UnsafePointer<Float>, frames: Int, channels: Int) -> Data {
        var out = Data(capacity: Int(Double(frames) / ratio + 2) * 2)
        for i in 0..<frames {
            var s: Float = 0
            for c in 0..<channels { s += ptr[i * channels + c] }
            lp += alpha * (s / Float(channels) - lp)
            while pos <= 1 {  // output samples between the last input sample and this one
                let v = last + (lp - last) * Float(pos)
                var q = Int16(max(-1, min(1, v)) * 32767).littleEndian
                withUnsafeBytes(of: &q) { out.append(contentsOf: $0) }
                pos += ratio
            }
            pos -= 1
            last = lp
        }
        return out
    }
}

final class Tap {
    var tapID = AudioObjectID(kAudioObjectUnknown), aggID = AudioObjectID(kAudioObjectUnknown)
    var procID: AudioDeviceIOProcID?
    let processes: [AudioObjectID], muted: Bool

    init?(processes: [AudioObjectID], muted: Bool, sink: @escaping (Data) -> Void) {
        self.processes = processes
        self.muted = muted
        let desc = CATapDescription(stereoMixdownOfProcesses: processes)
        desc.uuid = UUID()
        desc.name = "THURSDAY music tap"
        desc.isPrivate = true
        desc.muteBehavior = muted ? .muted : .unmuted
        var status = AudioHardwareCreateProcessTap(desc, &tapID)
        guard status == noErr else { log("tap failed (\(status)): is System Audio Recording allowed?"); return nil }
        guard let fmt = readProp(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()).first else {
            log("no tap format"); destroy(); return nil
        }
        var agg: [String: Any] = [
            kAudioAggregateDeviceNameKey: "THURSDAY music tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: desc.uuid.uuidString]],
        ]
        if let out = defaultOutputUID() {  // a clock for the aggregate
            agg[kAudioAggregateDeviceMainSubDeviceKey] = out
            agg[kAudioAggregateDeviceSubDeviceListKey] = [[kAudioSubDeviceUIDKey: out]]
        }
        status = AudioHardwareCreateAggregateDevice(agg as CFDictionary, &aggID)
        guard status == noErr else { log("aggregate failed (\(status))"); destroy(); return nil }
        let conv = Converter(inRate: fmt.mSampleRate)
        let channels = Int(max(1, fmt.mChannelsPerFrame))
        status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, nil) { _, input, _, _, _ in
            let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
            guard let buf = list.first, let data = buf.mData else { return }
            let frames = Int(buf.mDataByteSize) / (MemoryLayout<Float>.size * Int(max(1, buf.mNumberChannels)))
            sink(conv.convert(data.assumingMemoryBound(to: Float.self), frames: frames, channels: Int(buf.mNumberChannels)))
        }
        guard status == noErr, AudioDeviceStart(aggID, procID) == noErr else { log("start failed"); destroy(); return nil }
        log("tapping \(processes.count) process(es) at \(Int(fmt.mSampleRate)) Hz x\(channels), Mac \(muted ? "muted" : "audible")")
    }

    func destroy() {
        if aggID != kAudioObjectUnknown {
            AudioDeviceStop(aggID, procID)
            if let p = procID { AudioDeviceDestroyIOProcID(aggID, p) }
            AudioHardwareDestroyAggregateDevice(aggID)
        }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        aggID = AudioObjectID(kAudioObjectUnknown); tapID = AudioObjectID(kAudioObjectUnknown)
    }
}

final class Link: NSObject, URLSessionWebSocketDelegate {
    let token: String
    var task: URLSessionWebSocketTask?
    var open = false
    var muted = true
    var pending = Data()
    let q = DispatchQueue(label: "music-tap.send")
    var onMute: ((Bool) -> Void)?

    init(token: String) { self.token = token }

    func connect() {
        var req = URLRequest(url: URL(string: coreURL)!)
        req.setValue(token, forHTTPHeaderField: "X-Music-Tap-Token")
        task = URLSession(configuration: .default, delegate: self, delegateQueue: nil).webSocketTask(with: req)
        task?.resume()
        receive()
    }

    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol p: String?) {
        q.async { self.open = true }
        log("connected to core")
    }

    func urlSession(_ s: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith c: URLSessionWebSocketTask.CloseCode,
                    reason: Data?) { retry() }

    func urlSession(_ s: URLSession, task t: URLSessionTask, didCompleteWithError e: Error?) { if e != nil { retry() } }

    func retry() {
        q.async {
            guard self.task != nil else { return }
            self.open = false; self.task = nil
            DispatchQueue.global().asyncAfter(deadline: .now() + 2) { self.connect() }
        }
    }

    func receive() {
        task?.receive { [weak self] result in
            guard let self, case .success(let msg) = result else { return }
            if case .string(let text) = msg, let d = text.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any], let m = obj["mute"] as? Bool {
                self.onMute?(m)
            }
            self.receive()
        }
    }

    /// From the audio thread: buffer, and send whole 40 ms frames.
    func feed(_ pcm: Data) {
        q.async {
            guard self.open, let task = self.task else { self.pending.removeAll(); return }
            self.pending.append(pcm)
            while self.pending.count >= frameBytes {
                let frame = self.pending.prefix(frameBytes)
                self.pending.removeFirst(frameBytes)
                task.send(.data(Data(frame))) { _ in }
            }
            if self.pending.count > frameBytes * 25 { self.pending.removeAll() }  // a stalled socket: don't pile up
        }
    }
}

// --- main -------------------------------------------------------------------------------------

let tokenPath = NSString(string: "~/.config/thursday/music_tap_token").expandingTildeInPath
guard let token = try? String(contentsOfFile: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
      !token.isEmpty else { log("no token at \(tokenPath)"); exit(1) }

/// macOS asks for "System Audio Recording" only when a tap is made, and until it's allowed the
/// process list hides other apps' audio: so make (and drop) a throwaway tap once, up front.
func askPermission() {
    let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
    desc.isPrivate = true
    var id = AudioObjectID(kAudioObjectUnknown)
    let status = AudioHardwareCreateProcessTap(desc, &id)
    if status == noErr { AudioHardwareDestroyProcessTap(id) }
    log(status == noErr ? "System Audio Recording: allowed" : "System Audio Recording: not allowed yet (\(status))")
}

askPermission()
let link = Link(token: token)
var tap: Tap?
var wantMuted = true
let control = DispatchQueue(label: "music-tap.control")

var lastSeen = -1
func refresh() {
    let procs = appAudioProcesses().sorted()
    if procs.count != lastSeen {
        lastSeen = procs.count
        log("YouTube Music audio processes: \(procs.count)")
    }
    if let t = tap, t.processes == procs, t.muted == wantMuted { return }
    tap?.destroy(); tap = nil
    if !procs.isEmpty { tap = Tap(processes: procs, muted: wantMuted) { link.feed($0) } }
}

link.onMute = { m in control.async { wantMuted = m; refresh() } }
link.connect()
let timer = DispatchSource.makeTimerSource(queue: control)
timer.schedule(deadline: .now(), repeating: 3)  // the app's helper processes come and go
timer.setEventHandler { refresh() }
timer.resume()
signal(SIGTERM) { _ in tap?.destroy(); exit(0) }
dispatchMain()
