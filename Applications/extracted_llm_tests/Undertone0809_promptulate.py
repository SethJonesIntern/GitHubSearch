# Undertone0809/promptulate
# 1 LLM-backed test functions across 28 test files
# Source: https://github.com/Undertone0809/promptulate

# --- tests/old/tools/test_langchain_tools.py ---

def test_read_file():
    working_directory = TemporaryDirectory()

    lc_write_tool = FileManagementToolkit(
        root_dir=str(working_directory.name),
        selected_tools=["write_file"],
    ).get_tools()[0]

    tool = LangchainTool(lc_write_tool)
    tool.run({"file_path": "example.txt", "text": "Hello World!"})

    assert os.path.exists(os.path.join(working_directory.name, "example.txt"))

