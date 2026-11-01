# THURSDAY's deployer (deploy/deployer.sh): the one container with the Docker socket. It rebuilds
# thursday-core when Daniel taps Deploy (improvements.deploy_request), checks it, rolls back if not.
FROM docker:29.9.1-cli
RUN apk add --no-cache git curl
COPY deploy/deployer.sh /deployer.sh
ENTRYPOINT ["sh", "/deployer.sh"]
