from .agent import create_paper_request


def main():
    print("AI Question Paper Agent")
    print("-----------------------")

    user_input = input("\nTeacher: ")

    result = create_paper_request(user_input)

    print("\nAgent Output:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()