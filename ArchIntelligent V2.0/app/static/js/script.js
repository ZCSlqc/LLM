// 格式化消息文本
function formatMessage(text) {
    if (!text) return '';
    let lines = text.split('\n');
    let formattedLines = lines.map(line => {
        line = line.replace(/\*\*(.*?)\*\*/g, '<span class="bold-text">$1</span>');
        return line;
    });
    let processedText = formattedLines.join('\n');
    let sections = processedText
        .split('###')
        .filter(section => section.trim())
        .map(section => {
            let lines = section.split('\n').filter(line => line.trim());
            if (lines.length === 0) return '';
            let result = '';
            let currentIndex = 0;
            while (currentIndex < lines.length) {
                let line = lines[currentIndex].trim();
                if (/^\d+\./.test(line)) {
                    result += `<p class="section-title">${line}</p>`;
                } else if (line.startsWith('-')) {
                    result += `<p class="subsection"><span class="bold-text">${line.replace(/^-/, '').trim()}</span></p>`;
                } else if (line.includes(':')) {
                    let [subtitle, content] = line.split(':').map(part => part.trim());
                    result += `<p><span class="subtitle">${subtitle}</span>: ${content}</p>`;
                } else {
                    result += `<p>${line}</p>`;
                }
                currentIndex++;
            }
            return result;
        });
    return sections.join('');
}

let chatHistory = JSON.parse(sessionStorage.getItem('chatHistory') || '[]');

function saveHistory() {
    sessionStorage.setItem('chatHistory', JSON.stringify(chatHistory));
}

function clearChat() {
    document.getElementById('messages').innerHTML = '';
    chatHistory = [];
    saveHistory();
}

function displayMessage(role, message) {
    const messagesContainer = document.getElementById('messages');
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    const avatar = document.createElement('img');
    avatar.src = role === 'user' ? '/static/img/user-avatar.png' : '/static/img/bot-avatar.png';
    avatar.alt = role === 'user' ? 'User' : 'Bot';
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = role === 'user' ? message : formatMessage(message);
    messageElement.appendChild(avatar);
    messageElement.appendChild(messageContent);
    messagesContainer.appendChild(messageElement);
    messageElement.scrollIntoView({ behavior: 'smooth' });
}

function renderHistory() {
    document.getElementById('messages').innerHTML = '';
    for (const msg of chatHistory) {
        displayMessage(msg.role, msg.content);
    }
}

function sendMessage() {
    const inputElement = document.getElementById('chat-input');
    const message = inputElement.value;
    if (!message.trim()) return;
    displayMessage('user', message);
    chatHistory.push({ role: 'user', content: message });
    saveHistory();
    inputElement.value = '';
    const loadingElement = document.getElementById('loading');
    if (loadingElement) loadingElement.style.display = 'block';
    fetch('/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: chatHistory })
    })
    .then(res => res.json())
    .then(data => {
        console.log('后端返回：', data);
        displayMessage('bot', data.response);
        chatHistory.push({ role: 'assistant', content: data.response });
        saveHistory();
        if (loadingElement) loadingElement.style.display = 'none';
    })
    .catch(error => {
        if (loadingElement) loadingElement.style.display = 'none';
        displayMessage('bot', '出错了，请稍后再试。');
        console.error('Error:', error);
    });
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const chatContainer = document.querySelector('.chat-container');
    const messages = document.querySelector('.messages');
    chatContainer.classList.toggle('dark-mode');
    messages.classList.toggle('dark-mode');
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
}

document.addEventListener('DOMContentLoaded', () => {
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        document.querySelector('.chat-container').classList.add('dark-mode');
        document.querySelector('.messages').classList.add('dark-mode');
    }
    renderHistory();
    document.getElementById('chat-input').addEventListener('keypress', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
});

function toggleDropdown(event) {
    event.preventDefault();
    document.getElementById('dropdownMenu').classList.toggle('show');
}

window.onclick = function(event) {
    if (!event.target.matches('.dropdown button')) {
        const dropdowns = document.getElementsByClassName('dropdown-content');
        for (const dropdown of dropdowns) {
            if (dropdown.classList.contains('show')) {
                dropdown.classList.remove('show');
            }
        }
    }
} 