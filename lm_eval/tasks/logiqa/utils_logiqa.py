# Adapted for scriptless mirror `lucasmccabe/logiqa`
# Fields: context (str), query (str), options (list[str]), correct_option (int)
def doc_to_text(doc) -> str:
    choices = ["a", "b", "c", "d"]
    question = doc.get("query", doc.get("question", ""))
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + question + "\nChoices:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    # 新镜像用 correct_option(int)，旧版用 label(a/b/c/d)
    if "correct_option" in doc:
        return int(doc["correct_option"])
    choices = ["a", "b", "c", "d"]
    return choices.index(doc["label"].strip())
