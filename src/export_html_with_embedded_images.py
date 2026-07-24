import os
import base64
import re

def build_standalone_html():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_path = os.path.join(base_dir, "Predictive_RL_DVFS_Research_Paper.md")
    img_path = os.path.join(base_dir, "results", "benchmark_performance_comparison.png")
    out_html = os.path.join(base_dir, "IEEE_Predictive_RL_DVFS_Research_Paper.html")
    artifact_html = os.path.join(r"C:\Users\saisr\.gemini\antigravity-ide\brain\92f7ea13-7b89-4863-ac91-5f0810687e77", "IEEE_Predictive_RL_DVFS_Research_Paper.html")

    # Read base64 image
    with open(img_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    img_data_uri = f"data:image/png;base64,{b64_img}"

    # Read markdown paper
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert Markdown to HTML elements manually or via regex/formatting
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
    <title>Predictive RL-Based DVFS Governor for Batteryless Intermittent IoT Edge Nodes</title>
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
            font-size: 24pt;
            text-align: center;
            font-weight: bold;
            margin-bottom: 15px;
            color: #0b2545;
        }}
        .author-block {{
            text-align: center;
            font-size: 11pt;
            margin-bottom: 30px;
            color: #333;
        }}
        .abstract-box {{
            background: #f8f9fa;
            border-left: 4px solid #0b2545;
            padding: 18px;
            margin-bottom: 30px;
            font-size: 10.5pt;
            text-align: justify;
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
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background-color: #0b2545;
            color: #fff;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
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
        }}
        code {{
            font-family: 'Consolas', monospace;
            background-color: #f1f5f9;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        blockquote {{
            border-left: 3px solid #94a3b8;
            margin: 0;
            padding-left: 15px;
            color: #475569;
        }}
    </style>
</head>
<body>
"""

    # Format Markdown body to basic HTML tags
    body_html = md_text

    # Replace headings
    body_html = re.sub(r"^# (.*?)$", r"h1.\1", body_html, flags=re.MULTILINE)
    body_html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", body_html, flags=re.MULTILINE)
    body_html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", body_html, flags=re.MULTILINE)

    # Replace bold & italic
    body_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", body_html)
    body_html = re.sub(r"\*(.*?)\*", r"<i>\1</i>", body_html)

    # Code blocks
    body_html = re.sub(r"```(.*?)\n(.*?)```", r"<pre><code>\2</code></pre>", body_html, flags=re.DOTALL)

    # Paragraphs and line breaks
    paragraphs = body_html.split("\n\n")
    formatted_paragraphs = []
    for p in paragraphs:
        p_strip = p.strip()
        if p_strip.startswith("<h2>") or p_strip.startswith("<h3>") or p_strip.startswith("<pre>") or p_strip.startswith("<div") or p_strip.startswith("h1.") or p_strip.startswith("|"):
            if p_strip.startswith("h1."):
                formatted_paragraphs.append(f"<h1>{p_strip[3:]}</h1>")
            else:
                formatted_paragraphs.append(p_strip)
        else:
            formatted_paragraphs.append(f"<p>{p_strip}</p>")

    html_content += "\n".join(formatted_paragraphs)
    html_content += "\n</body>\n</html>"

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    if os.path.exists(os.path.dirname(artifact_html)):
        with open(artifact_html, "w", encoding="utf-8") as f:
            f.write(html_content)

    print(f"Successfully generated HTML with embedded base64 image at: {out_html}")

if __name__ == "__main__":
    build_standalone_html()
