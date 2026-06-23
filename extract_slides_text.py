import fitz

files = {
    'predefense': r'C:\Users\danil\Downloads\Предзащита презентация.pptx.pdf',
    'kadyrova': r'C:\Users\danil\Downloads\Telegram Desktop\ВКР_КадыроваЭлина.pptx (3).pdf',
}
for name, path in files.items():
    print('\n\n##################################################')
    print('FILE:', name)
    print('##################################################')
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        print('\n===== SLIDE', i + 1, '=====')
        print(text)
    doc.close()
