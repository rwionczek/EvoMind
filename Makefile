start:
	docker compose run --rm --build app python main.py

tensorboard:
	tensorboard --logdir runs
