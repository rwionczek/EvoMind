start:
	docker-compose run --rm app npm start

test:
	docker-compose run --rm app ./node_modules/mocha/bin/mocha

shell:
	docker-compose run --rm app bash
