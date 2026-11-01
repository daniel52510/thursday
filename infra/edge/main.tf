# THURSDAY's public edge (AWS slice A2): one small Lightsail server with a fixed IP. It will
# run Caddy (HTTPS, path allowlist) and the WireGuard end of a tunnel that the Mac opens
# outbound, so the house never opens a port. The brain stays on the Mac.
#
#   export AWS_PROFILE=thursday TF_VAR_home_ip=$(curl -s https://checkip.amazonaws.com)
#   terraform init && terraform plan
#
# Ports are opened slice by slice: SSH (home IP + Lightsail's browser console) and WireGuard
# now; 80/443 only in A4, once Caddy is there to answer them.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
}

provider "aws" {
  region = "us-east-1" # the deployer is locked to this region
}

variable "home_ip" {
  description = "Your home's public IPv4: the only address allowed to SSH in. From TF_VAR_home_ip, never committed."
  type        = string
  validation {
    condition     = can(regex("^\\d+\\.\\d+\\.\\d+\\.\\d+$", var.home_ip))
    error_message = "home_ip must be a plain IPv4 address."
  }
}

variable "ssh_public_key" {
  description = "Public key for SSH into the edge."
  type        = string
  default     = "~/.ssh/thursday_edge.pub"
}

variable "wireguard_port" {
  type    = number
  default = 51820
}

locals {
  tags = { project = "thursday", slice = "aws-edge" }
}

resource "aws_lightsail_key_pair" "edge" {
  name       = "thursday-edge"
  public_key = file(pathexpand(var.ssh_public_key))
  tags       = local.tags
}

resource "aws_lightsail_instance" "edge" {
  name              = "thursday-edge"
  availability_zone = "us-east-1a"
  blueprint_id      = "ubuntu_24_04"
  bundle_id         = "nano_3_0" # $5/mo: 512 MB, 2 vCPU, 20 GB, 1 TB transfer, public IPv4
  key_pair_name     = aws_lightsail_key_pair.edge.name
  # IPv4 only: with dual-stack, each port rule also gets an IPv6 range that defaults to open-to-all,
  # which would put SSH on the open internet over IPv6. Nothing here needs IPv6.
  ip_address_type = "ipv4"
  # First boot: security updates applied automatically from day one. Caddy and WireGuard come in A3/A4.
  user_data = <<-EOT
    #!/bin/bash
    set -e
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
    DEBIAN_FRONTEND=noninteractive apt-get -y install unattended-upgrades
    systemctl enable --now unattended-upgrades
  EOT
  tags      = local.tags
}

resource "aws_lightsail_static_ip" "edge" {
  name = "thursday-edge-ip" # free while attached to a running instance
}

resource "aws_lightsail_static_ip_attachment" "edge" {
  static_ip_name = aws_lightsail_static_ip.edge.name
  instance_name  = aws_lightsail_instance.edge.name
}

# Replaces Lightsail's defaults (SSH and HTTP open to the world) with exactly these.
resource "aws_lightsail_instance_public_ports" "edge" {
  instance_name = aws_lightsail_instance.edge.name

  port_info {
    protocol          = "tcp"
    from_port         = 22
    to_port           = 22
    cidrs             = ["${var.home_ip}/32"]
    cidr_list_aliases = ["lightsail-connect"] # the console's browser SSH: a way back in if your home IP changes
  }

  port_info {
    # WireGuard answers nothing without a valid key, so an open UDP port reveals nothing.
    protocol  = "udp"
    from_port = var.wireguard_port
    to_port   = var.wireguard_port
    cidrs     = ["0.0.0.0/0"]
  }
}

output "edge_ip" {
  value = aws_lightsail_static_ip.edge.ip_address
}

output "ssh" {
  value = "ssh -i ~/.ssh/thursday_edge ubuntu@${aws_lightsail_static_ip.edge.ip_address}"
}
