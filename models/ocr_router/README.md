Put the trained OCR router artifacts here:

- line_candidate_selector.pkl
- blank_classifier.pkl

The live demo pipeline loads line_candidate_selector.pkl by default. blank_classifier.pkl is loaded for reporting but the blank gate is disabled by default in solution/pipeline.py because the private route used the line candidate selector as the main OCR selector.
