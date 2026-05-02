PYTHON = python
SRC = src

setup:
	pip install -r requirements.txt

collect_weather:
	$(PYTHON) $(SRC)/collect_weather.py --start 2025-01-01 --end 2026-03-31

clean:
	$(PYTHON) $(SRC)/clean_data.py

features:
	$(PYTHON) $(SRC)/features.py

train:
	$(PYTHON) $(SRC)/train.py

evaluate:
	$(PYTHON) $(SRC)/evaluate.py

visualize:
	$(PYTHON) $(SRC)/visualize.py

all: clean features train visualize evaluate

predict:
	$(PYTHON) $(SRC)/predict.py

test:
	pytest tests/ -v

reset:
	rm -rf data/processed/*