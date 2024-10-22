.PHONY: install

install:
	pip install -r requirements.txt

run:
	python -m spacy download es_core_news_sm
	python main.py