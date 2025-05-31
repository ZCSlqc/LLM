import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from app.core.knowledge_base import KnowledgeBase
from app.utils.config import Config

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'), static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))
app.secret_key = 'pdfchat-secret-key'

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化知识库
knowledge_base = KnowledgeBase()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data_json = request.get_json()
        user_input = data_json.get('message')
        history = data_json.get('history', [])
        # 限制history长度，避免token超限
        max_history = 10
        if len(history) > max_history:
            history = history[-max_history:]
        # 构造完整messages
        messages = [{"role": "system", "content": "你是专业的知识问答助手。"}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_input})

        # 拼接成字符串
        query_str = ""
        for msg in messages:
            if msg["role"] == "system":
                query_str += f"系统：{msg['content']}\n"
            elif msg["role"] == "user":
                query_str += f"用户：{msg['content']}\n"
            elif msg["role"] == "assistant":
                query_str += f"助手：{msg['content']}\n"

        try:
            result = knowledge_base.query(query_str)
            answer = result.get('response', '')
            # citations = result.get('citations', [])
        except Exception as e:
            answer = f"调用大模型出错: {e}"
        return jsonify({'response': answer})
    else:
        # 这里加上 GET 请求的返回
        return render_template('index.html')

@app.route('/RAG', methods=['GET', 'POST'])
def RAG():
    status = knowledge_base.get_status()
    if request.method == 'POST':
        if 'pdf' in request.files:
            files = request.files.getlist('pdf')
            for file in files:
                if file.filename:
                    filename = secure_filename(file.filename)
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    try:
                        file.save(save_path)
                        knowledge_base.add_pdf_document(save_path)
                    except Exception as e:
                        flash(f"文件 {filename} 保存或添加失败: {e}")
            flash('文件上传成功！')
    return render_template('RAG.html', status=status)

@app.route('/clear', methods=['POST'])
def clear_knowledge_base():
    # 1. 清空知识库
    knowledge_base.clear_knowledge_base()
    # 2. 清空 uploads 文件夹
    uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    uploads_dir = os.path.abspath(uploads_dir)
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"删除文件失败: {file_path}, 错误: {e}")
    flash('知识库已清空')
    return redirect(url_for('RAG'))

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(knowledge_base.get_status())

if __name__ == '__main__':
    app.run(host=Config.APP_HOST, port=Config.APP_PORT, debug=True)