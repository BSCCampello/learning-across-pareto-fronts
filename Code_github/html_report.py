class HtmlReport:

    def __init__(self, caminho):

        self.caminho = caminho
        self.linhas = ["<html>", "<body>"]

    def add_heading(self, texto, level=1):

        self.linhas.append(f"<h{level}>{texto}</h{level}>")

    def add_text(self, texto):

        self.linhas.append(f"<p>{texto}</p>")

    def save(self):

        self.linhas.extend(["</body>", "</html>"])

        with open(self.caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas))

    def add_hr(self):
        self.linhas.append("<hr>")


    def add_image(self, caminho, largura=800):
        self.linhas.append(f'<img src="{caminho}" width="{largura}"><br>')


    def add_table(self, headers, rows):
        html = "<table border='1' cellpadding='5' cellspacing='0'>"
        html += "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

        for row in rows:
            html += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"

        html += "</table><br>"

        self.linhas.append(html)