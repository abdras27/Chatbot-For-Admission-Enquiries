import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input })
      });
      const data = await response.json();
      const botMsg = {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot'
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error('Bot error:', err);
      setMessages(prev => [...prev, {
        id: Date.now() + 2,
        text: 'Error: could not get response.',
        sender: 'bot'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-white-100">
      <div className="flex flex-col flex-1 justify-between w-full max-w-md mx-auto bg-white shadow-lg">
        
        {/* Chat messages */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start space-x-2 ${msg.sender === "bot" ? "justify-start" : "justify-end"}`}
            >
              {msg.sender === "bot" && (
                <div className="flex-shrink-0">
                  <span className="flex items-center justify-center bg-blue-500 text-white rounded-full p-2 text-2xl">🤖</span>
                </div>
              )}

              <div className={`
                px-4 py-2 rounded-lg max-w-[75%] break-words
                ${msg.sender === "bot" ? "bg-blue-100 text-black" : "bg-green-100 text-black"}
              `}>
                {msg.text}
              </div>

              {msg.sender === "user" && (
                <div className="flex-shrink-0">
                  <span className="flex items-center justify-center bg-green-500 text-white rounded-full p-2 text-2xl">🧑‍💻</span>
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="px-4 py-2 rounded-lg bg-blue-100 text-black flex items-center">
                <div className="animate-spin h-5 w-5 border-4 border-blue-500 border-t-transparent rounded-full"></div>
                <span className="ml-2">Typing...</span>
              </div>
            </div>
          )}

          <div ref={bottomRef}></div>
        </div>

        {/* Input area */}
        <form onSubmit={handleSend} className="p-4 border-t bg-gray-50 flex">
          <input
            type="text"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 rounded-r-lg"
            disabled={isLoading || !input.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
