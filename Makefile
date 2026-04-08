# CVA-SACS v6
.PHONY: setup run test clean

setup:
	pip install -r requirements.txt

run:
	streamlit run cva_sacs_v6.py

test:
	python -m pytest tests/ -v --tb=short

clean:
	rm -rf models_v6/ __pycache__ .pytest_cache
