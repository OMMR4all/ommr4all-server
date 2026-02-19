if __name__ == '__main__':
    import django

    os.environ['DJANGO_SETTINGS_MODULE'] = 'ommr4all.settings'
    django.setup()
    # get all json files from directory with json endings
    rendered_json_files = {}
    for file in os.listdir(dataset_json_file_path):
        if file.endswith(".json"):
            rendered_json_files[file.replace(".json", "")] = file
            #print(file)
    # get all books from database
    book = DatabaseBook("Graduel_Syn2")
    pages = book.pages()
    # pages = [pages[5]]  # 0:45
    # for each book get all pages and compare with json files
    excepted_ids = []
    for page in pages:
        page_id = page.page
        #print(page_id)
        if page_id not in rendered_json_files.keys():
            print("Page " + page_id + " not in rendered files")
        if page_id not in ["0388", "1012"]:
            continue
        # remove_upper_regions(page.pcgts())
        # set_document_start(page.pcgts())
        # assign_text_to_lines(page, rendered_json_files)
        # assign_syllabels_to_symbols2(page, rendered_json_files, book)
        try:
            fill_meta_infos_to_docs(book, page, rendered_json_files)
        except Exception as e:
            print(page_id)
            excepted_ids.append(page_id)
        # page.pcgts()
        # pcgts = page.pcgts().page.all_text_lines()
    print("Faulty_pages")
    for i in excepted_ids:
        print(i)