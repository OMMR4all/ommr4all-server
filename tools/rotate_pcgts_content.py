from database.database_book import DatabaseBook

if __name__ == "__main__":
    book = DatabaseBook('Graduel_Fully_Annotated')
    page = book.page('Graduel_de_leglise_de_Nevers_035')
    pcgts = page.pcgts()
    pcgts.page.rotate(-0.125)
    pcgts.to_file(page.file('pcgts').local_path())
