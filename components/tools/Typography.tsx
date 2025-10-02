
import React, { useState, useCallback } from 'react';
import { generateText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface TypographyProps {
    tool: Tool;
    onBack: () => void;
}

const Typography: React.FC<TypographyProps> = ({ tool, onBack }) => {
    const [mood, setMood] = useState('');
    const [style, setStyle] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!mood.trim() || !style.trim()) {
            setError('Please provide both mood and font style.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `I need a font pairing for a project with a '${mood}' mood. The primary headline font should feel ${style}. Suggest a complementary body font from Google Fonts. Explain why the pairing works well together. Provide the names of the fonts clearly.`;
            const response = await generateText(prompt);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [mood, style]);
    
    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe el 'mood' o la sensación que quieres transmitir y el estilo de fuente que buscas para los títulos. La IA te sugerirá un par de fuentes (título y cuerpo) que combinan bien.
                </p>
            </div>

            <div className="space-y-4 bg-gray-800/50 p-6 rounded-lg">
                <input type="text" value={mood} onChange={(e) => setMood(e.target.value)} placeholder="Project Mood (e.g., elegant, modern, playful)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                <input type="text" value={style} onChange={(e) => setStyle(e.target.value)} placeholder="Desired Headline Font Style (e.g., a bold serif, a clean sans-serif)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Buscando Pares...' : 'Sugerir Pares de Fuentes'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Consultando al tipógrafo IA...</p>}
                {result && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Sugerencia de Pairing</h3>
                        <div className="bg-gray-800 p-6 rounded-lg">
                           <pre className="whitespace-pre-wrap text-left bg-gray-900 p-4 rounded font-sans text-gray-300">{result}</pre>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Typography;
