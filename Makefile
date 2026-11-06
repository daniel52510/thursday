# Everyday commands. `make setup` once on a fresh Mac, then `make up`.
.PHONY: setup up down restart logs test sat-deploy sat-report

setup:          ## first-time install (scripts/setup.sh)
	scripts/setup.sh

up:             ## build and start the containers
	docker compose up -d --build

down:           ## stop the containers (data in brain/ stays)
	docker compose down

restart:        ## rebuild core only
	docker compose up -d --build thursday-core

logs:           ## follow core's log
	docker compose logs -f thursday-core

test:           ## run the test suite (in the runner image: no network, repo read-only)
	scripts/test.sh

sat-deploy:     ## copy the satellite to the Pi (needs THURSDAY_SAT_SSH, THURSDAY_SAT_KEY)
	scripts/deploy_sat.sh

sat-report:     ## the satellite's tuning numbers from its logs
	scripts/sat_report.sh
