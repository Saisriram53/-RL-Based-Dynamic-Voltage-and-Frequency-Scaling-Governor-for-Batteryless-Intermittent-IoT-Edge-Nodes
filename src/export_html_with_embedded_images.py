import os
import base64
import re
import html

def parse_markdown_table(table_lines):
    """Parses markdown table lines into clean HTML table syntax."""
    if len(table_lines) < 2:
        return "\n".join(table_lines)
    
    headers = [cell.strip() for cell in table_lines[0].strip('|').split('|')]
    # line 1 is separator |---|---|
    rows = []
    for line in table_lines[2:]:
        if '|' in line:
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            rows.append(cells)
            
    html_out = ['<table class="ieee-table">', '  <thead>', '    <tr>']
    for h in headers:
        html_out.append(f'      <th>{h}</th>')
    html_out.extend(['    </tr>', '  </thead>', '  <tbody>'])
    for r in rows:
        html_out.append('    <tr>')
        for c in r:
            html_out.append(f'      <td>{c}</td>')
        html_out.append('    </tr>')
    html_out.extend(['  </tbody>', '</table>'])
    return '\n'.join(html_out)

def build_standalone_html():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(base_dir, "Predictive_RL_DVFS_Research_Paper.md")
    img_path = os.path.join(base_dir, "results", "benchmark_performance_comparison.png")
    out_html = os.path.join(base_dir, "IEEE_Predictive_RL_DVFS_Research_Paper.html")

    # Read base64 image
    with open(img_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{b64_img}"

    # Read markdown paper
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Extract paper title from # title line
    title_match = re.search(r"^# (.*?)$", md_text, flags=re.MULTILINE)
    paper_title = title_match.group(1).strip() if title_match else "Gradient-Aware RL-Based DVFS Governor"

    # Replace markdown image links with base64 data URI img tag
    md_text = re.sub(
        r"!\[.*?\]\(.*?\)",
        f'<div style="text-align: center; margin: 25px 0;"><img src="{img_data_uri}" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" /><p style="font-size: 0.9em; color: #555; margin-top: 8px;"><b>Figure 1:</b> Multi-panel transient dynamics comparison across Always-Max, Powersave, Static Threshold, PPO RL, and DQN governors.</p></div>',
        md_text
    )

    # HTML template with embedded IEEE CSS styles & MathJax for LaTeX
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{paper_title}</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif, 'Helvetica Neue', Arial;
            line-height: 1.6;
            color: #111;
            max-width: 960px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
        }}
        h1 {{
            font-size: 22pt;
            text-align: center;
            font-weight: bold;
            margin-bottom: 15px;
            color: #0b2545;
            line-height: 1.3;
        }}
        .author-block {{
            text-align: center;
            font-size: 11pt;
            margin-bottom: 30px;
            color: #333;
        }}
        h2 {{
            font-size: 14pt;
            color: #0b2545;
            border-bottom: 2px solid #0b2545;
            padding-bottom: 4px;
            margin-top: 30px;
            text-transform: uppercase;
        }}
        h3 {{
            font-size: 12pt;
            color: #134074;
            margin-top: 20px;
        }}
        table.ieee-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }}
        table.ieee-table th, table.ieee-table td {{
            border: 1px solid #cbd5e1;
            padding: 10px 12px;
            text-align: left;
        }}
        table.ieee-table th {{
            background-color: #0b2545;
            color: #fff;
            font-weight: bold;
        }}
        table.ieee-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        pre {{
            background-color: #1e293b;
            color: #f8fafc;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 9.5pt;
            line-height: 1.4;
        }}
        code {{
            font-family: 'Consolas', monospace;
            background-color: #f1f5f9;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            color: inherit;
        }}
        blockquote {{
            border-left: 3px solid #94a3b8;
            margin: 0;
            padding-left: 15px;
            color: #475569;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
"""

    lines = md_text.splitlines()
    body_html_lines = []
    in_code_block = False
    code_block_lines = []
    in_table = False
    table_lines = []

    for line in lines:
        # Code blocks toggle
        if line.startswith("```"):
            if in_code_block:
                escaped_code = html.escape("\n".join(code_block_lines))
                body_html_lines.append(f"<pre><code>{escaped_code}</code></pre>")
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_block_lines = []
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Tables toggle
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            continue
        else:
            if in_table:
                table_html = parse_markdown_table(table_lines)
                body_html_lines.append(table_html)
                table_lines = []
                in_table = False

        # Headings
        if line.startswith("# "):
            body_html_lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            body_html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            body_html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("---"):
            body_html_lines.append("<hr />")
        else:
            # Inline replacements
            l_fmt = line
            l_fmt = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", l_fmt)
            l_fmt = re.sub(r"\*(.*?)\*", r"<i>\1</i>", l_fmt)
            l_fmt = re.sub(r"`(.*?)`", r"<code>\1</code>", l_fmt)

            if l_fmt.strip():
                if not l_fmt.strip().startswith("<"):
                    body_html_lines.append(f"<p>{l_fmt.strip()}</p>")
                else:
                    body_html_lines.append(l_fmt.strip())

    if in_table:
        table_html = parse_markdown_table(table_lines)
        body_html_lines.append(table_html)

    html_content += "\n".join(body_html_lines)
    html_content += "\n</body>\n</html>"

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully generated HTML with embedded base64 image at: {out_html}")

if __name__ == "__main__":
    build_standalone_html()
