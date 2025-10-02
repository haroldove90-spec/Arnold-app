
import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';
import { Type } from '@google/genai';

interface SocialMediaOptimizationProps {
    tool: Tool;
    onBack: () => void;
}

const platforms = [
    "Instagram Post", "Instagram Story", "Facebook Post", "Facebook Cover",
    "Twitter Post", "LinkedIn Post", "TikTok Video", "YouTube Thumbnail"
];

const SocialMediaOptimization: React.FC<SocialMediaOptimizationProps> = ({ tool, onBack }) => {
    const [platform, setPlatform] = useState(platforms[0]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);

    const handleGenerate = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `Provide the ideal image/video specifications and best practices for a '${platform}'. Include tips for maximizing engagement.`;
            const schema = {
                type: Type.OBJECT,
                properties: {
                    platform: { type: Type.STRING },
                    dimensions: { type: Type.STRING, description: "e.g., 1080x1080 pixels" },
                    aspectRatio: { type: Type.STRING, description: "e.g., 1:1" },
                    formats: { type: Type.ARRAY, items: { type: Type.STRING }},
                    bestPractices: { type: Type.ARRAY, items: { type: Type.STRING }}
                },
                required: ["platform", "dimensions", "aspectRatio", "formats", "bestPractices"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [platform]);
    
    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Elige una plataforma de redes sociales de la lista. La IA te proporcionará las dimensiones exactas, formatos recomendados y mejores prácticas de diseño para esa plataforma.
                </p>
            </div>

            <div className="flex space-x-2">
                <select value={platform} onChange={(e) => setPlatform(e.target.value)} className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none" disabled={isLoading}>
                    {platforms.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
                >
                    {isLoading ? <Spinner /> : 'Obtener Guía'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center">Generando guía de optimización...</p>}
                {result && (
                    <div className="bg-gray-800 p-6 rounded-lg">
                        <h3 className="text-2xl font-semibold mb-4 text-center">Guía para {result.platform}</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center mb-6">
                            <div><p className="text-sm text-gray-400">Dimensiones</p><p className="font-bold text-lg">{result.dimensions}</p></div>
                            <div><p className="text-sm text-gray-400">Aspect Ratio</p><p className="font-bold text-lg">{result.aspectRatio}</p></div>
                             <div><p className="text-sm text-gray-400">Formatos</p><p className="font-bold text-lg">{result.formats.join(', ')}</p></div>
                        </div>
                        <div>
                             <h4 className="text-xl font-bold text-purple-400">Mejores Prácticas</h4>
                             <ul className="list-disc list-inside mt-2 space-y-1 text-gray-300">
                                {result.bestPractices.map((tip: string, index: number) => <li key={index}>{tip}</li>)}
                             </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SocialMediaOptimization;
