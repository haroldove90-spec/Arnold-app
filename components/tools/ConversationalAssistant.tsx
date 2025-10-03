import React, { useState, useCallback, useRef, useEffect } from 'react';
import { continueConversation } from '../../services/geminiService';
import type { Message } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ConversationalAssistantProps {
    tool: Tool;
    onBack: () => void;
}

const ConversationalAssistant: React.FC<ConversationalAssistantProps> = ({ tool, onBack }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [currentMessage, setCurrentMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Scroll to bottom whenever messages change
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);
    
    useEffect(() => {
      // Set an initial message from the assistant
      setMessages([
        {
          role: 'model',
          text: "¡Hola, Arnold! Soy tu asistente de diseño. ¿En qué puedo ayudarte hoy? Puedes pedirme ideas, críticas o ayuda técnica."
        }
      ]);
    }, []);

    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!currentMessage.trim() || isLoading) return;

        const userMessage: Message = { role: 'user', text: currentMessage.trim() };
        const newMessages = [...messages, userMessage];
        
        setMessages(newMessages);
        setCurrentMessage('');
        setIsLoading(true);
        setError(null);

        try {
            const history = newMessages.slice(0, -1); // Exclude the latest user message for the history param
            const response = await continueConversation(history, userMessage.text);
            const modelMessage: Message = { role: 'model', text: response.text };
            setMessages(prev => [...prev, modelMessage]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
             // Optionally remove the user's message if the call fails
            setMessages(prev => prev.slice(0, -1));
        } finally {
            setIsLoading(false);
        }
    }, [currentMessage, isLoading, messages]);

    return (
        <div className="max-w-4xl mx-auto flex flex-col h-[80vh]">
            <div className="flex-shrink-0">
                <button onClick={onBack} className="mb-4 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
                <div className="text-center mb-4">
                    <h2 className="text-3xl font-bold">{tool.title}</h2>
                    <p className="text-gray-400 mt-2">{tool.description}</p>
                </div>
            </div>

            <div ref={chatContainerRef} className="flex-grow bg-gray-800/50 rounded-t-lg p-4 overflow-y-auto space-y-4">
                {messages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-lg p-3 rounded-lg ${msg.role === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-200'}`}>
                            <p className="whitespace-pre-wrap">{msg.text}</p>
                        </div>
                    </div>
                ))}
            </div>
            
            <div className="flex-shrink-0 bg-gray-800/50 rounded-b-lg p-4">
                {error && <p className="text-red-400 text-sm mb-2 text-center">{error}</p>}
                <form onSubmit={handleSubmit} className="flex items-center space-x-2">
                    <input
                        type="text"
                        value={currentMessage}
                        onChange={(e) => setCurrentMessage(e.target.value)}
                        placeholder="Escribe tu mensaje aquí..."
                        className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                        disabled={isLoading}
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !currentMessage.trim()}
                        className="flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold p-3 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                    >
                        {isLoading ? <Spinner /> : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                            </svg>
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
};

export default ConversationalAssistant;