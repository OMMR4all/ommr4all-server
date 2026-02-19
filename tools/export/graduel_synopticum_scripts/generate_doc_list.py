
def created_filtered_excel_doc_list_by_keyword(workbook, workbook2, filter_sheet, chat_column, filter_column, filter_key_word):
    """
    Create a filtered list of documents based on a keyword in a specified column.

    :param workbook: The workbook containing the document data.
    :param workbook2: The workbook to save the filtered results.
    :param chat_column: The column containing the document IDs.
    :param filter_column: The column to filter by keyword.
    :param filter_key_word: The keyword to filter documents.
    """
    sheet = workbook.get_sheet_by_name(filter_sheet)
    sheet2 = workbook2.active

    doc_list = []

    for i in range(2, 2500):
        doc_id = sheet[chat_column + str(i)].value
        if doc_id is None:
            continue
        #print(sheet[filter_column + str(i)].value)
        #print(filter_column + str(i))
        filter_value = sheet[filter_column + str(i)].value
        if filter_value is not None and str(filter_key_word) == str(filter_value).lower():
            doc_list.append(doc_id)

    for ind, i in enumerate(doc_list):
        sheet2["A" + str(ind + 3)] = i
    workbook2.save(f'/tmp/Graduale_{filter_sheet}_filtered.xlsx')
    return doc_list


if __name__ == "__main__":
    import openpyxl

    docs_sheet = "Symbols Docs"

    filter_sheet = "All_Docs_except_to_long"
    chant_column = "A"
    filter_column = "O"
    filter_key_word= "no"

    #filter_sheet = "No_Differences"
    #chant_column = "A"
    #filter_column = "O"
    #filter_key_word= "0"

    #filter_sheet = "Minor Differences"
    #chant_column = "A"
    #filter_column = "Q"
    #filter_key_word= "0"
    workbook = openpyxl.load_workbook('/tmp/Graduale_Results_05_04_2312.xlsx')
    sheet = workbook.active

    workbook2 = openpyxl.Workbook()

    created_filtered_excel_doc_list_by_keyword(workbook, workbook2, filter_sheet, chant_column, filter_column, filter_key_word)