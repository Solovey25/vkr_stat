import fitz, os, traceback

files = {
    'template': r'C:\Users\danil\Downloads\Вариант_шаблона_презентации.pptx.pdf',
    'predefense': r'C:\Users\danil\Downloads\Предзащита презентация.pptx.pdf',
    'kadyrova': r'C:\Users\danil\Downloads\Telegram Desktop\ВКР_КадыроваЭлина.pptx (3).pdf',
}
outdir = r'C:\git\vkr_stat\slides'
os.makedirs(outdir, exist_ok=True)
for name, path in files.items():
    try:
        doc = fitz.open(path)
        print(name, 'pages:', doc.page_count)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6))
            fn = os.path.join(outdir, name + '_' + str(i + 1).zfill(2) + '.png')
            pix.save(fn)
        doc.close()
    except Exception:
        traceback.print_exc()
print('done', len(os.listdir(outdir)), 'files')
