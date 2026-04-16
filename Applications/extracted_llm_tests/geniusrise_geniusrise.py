# geniusrise/geniusrise
# 6 LLM-backed test functions across 40 test files
# Source: https://github.com/geniusrise/geniusrise

# --- geniusrise/inference/text/tests/test_embeddings.py ---

def test_generate_sentence_transformer_embeddings(model_name):
    model = SentenceTransformer(model_name, device="cuda")
    _model = AutoModel.from_pretrained(model_name)
    sentences = ["This is a test sentence.", "Another test sentence."]
    embeddings = generate_sentence_transformer_embeddings(sentences=sentences, model=model)
    assert all(
        [
            x.shape[0] == _model.config.hidden_size or x.shape[0] == _model.config.max_position_embeddings
            for x in embeddings
        ]
    )

def test_generate_embeddings(model_name):
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence = "This is a test sentence."
    embeddings = generate_embeddings(sentence=sentence, model=model, tokenizer=tokenizer)
    assert embeddings.shape == (1, model.config.hidden_size)

def test_generate_contiguous_embeddings(model_name):
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence = "This is a test sentence."
    embeddings_list = generate_contiguous_embeddings(sentence=sentence, model=model, tokenizer=tokenizer)
    assert all(embeddings.shape == (1, model.config.hidden_size) for embeddings, _ in embeddings_list)

def test_generate_combination_embeddings(model_name):
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence = "This is a test sentence."
    embeddings_list = generate_combination_embeddings(sentence=sentence, model=model, tokenizer=tokenizer)
    assert all(embeddings.shape == (1, model.config.hidden_size) for embeddings, _ in embeddings_list)

def test_generate_permutation_embeddings(model_name):
    model = AutoModel.from_pretrained(model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence = "This is a test sentence."
    embeddings_list = generate_permutation_embeddings(sentence=sentence, model=model, tokenizer=tokenizer)
    assert all(embeddings.shape == (1, model.config.hidden_size) for embeddings, _ in embeddings_list)


# --- geniusrise/inference/vision/tests/imgclass/test_bulk.py ---

def test_prediction_file_creation_and_content(image_classification_bulk, image_dataset):
    dataset_path, ext = image_dataset
    image_classification_bulk.load_dataset(dataset_path)

    model_name = "microsoft/resnet-50"
    image_classification_bulk.classify(
        model_name=model_name,
        model_class="AutoModelForImageClassification",
        processor_class="AutoProcessor",
        device_map="cpu",
    )

    prediction_files = glob.glob(os.path.join(image_classification_bulk.output.output_folder, "*.json"))
    assert len(prediction_files) > 0, "No prediction files found in the output directory."

    with open(prediction_files[0], "r") as file:
        predictions = json.load(file)
        assert isinstance(predictions, list), "Predictions file should contain a list."
        assert len(predictions) > 0, "Prediction list is empty."

