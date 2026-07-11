import ast


def process_ast(string):
    return ast.literal_eval(string)


def _is_old_schema(doc):
    return "problems" in doc


def last_problem(doc):
    if _is_old_schema(doc):
        return process_ast(doc["problems"])[-1]
    return {
        "question": doc["question"],
        "options": doc["options"],
        "answer": doc["answer"],
    }


def get_answer_option(problem):
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    answer = letter_to_num[problem["answer"]]
    return problem["options"][answer]


def doc_to_choice(doc):
    problem = last_problem(doc)
    return [problem["options"][i] for i in range(4)]


def doc_to_text(doc):
    if _is_old_schema(doc):
        text = "Article: " + doc["article"] + "\n\n"
        for problem in process_ast(doc["problems"])[:-1]:
            if problem["question"][-6:] == "  _  .":
                text += problem["question"][-5:] + get_answer_option(problem) + "\n"
            else:
                question = "Question: " + problem["question"] + "\n"
                answer = "Answer: " + get_answer_option(problem) + "\n"
                text += question + answer
        text += last_problem(doc)["question"]
        return text

    # HF `race` schema has one question per row.
    return "Article: " + doc["article"] + "\n\nQuestion: " + doc["question"] + "\nAnswer:"


def doc_to_target(doc):
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    answer = letter_to_num[last_problem(doc)["answer"]]
    return answer
