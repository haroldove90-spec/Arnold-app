
import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';
import { Type } from '@google/genai';

interface TrendAnalysisProps {
    tool: Tool;
    onBack: () => void;
}

const TrendAnalysis: React.FC<TrendAnalysisProps> = ({ tool, onBack }) => {
    const [market, setMarket] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!market.trim()) {
            setError('Please enter a target market.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `As a design trend analyst, describe the current visual trends for the '${market}' target audience. Include popular color palettes (with hex codes), typography styles, imagery, and overall mood. Provide 3 concrete visual concepts.`;
            const schema = {
                type: Type.OBJECT,
                properties: {
                    summary: { type: Type.STRING, description: "Overall mood and trend summary." },
                    colorPalettes: { type: Type.ARRAY, items: { type: Type.STRING, description: "e.g., 'Vibrant Gradients: #FF00FF, #00FFFF'" }},
                    typography: { type: Type.ARRAY, items: { type: Type.STRING, description: "e.g., 'Bold Serifs: Playfair Display'" } },
                    imageryStyle: { type: Type.STRING },
                    concepts: {
                        type: Type.ARRAY,
                        items: {
                            type: Type.OBJECT,
                            properties: {
                                name: { type: Type.STRING },
                                description: { type: Type.STRING }
                            },
                             required: ["name", "description"]
                        }
                    }
                },
                required: ["summary", "colorPalettes", "typography", "imageryStyle", "concepts"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err)
 {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [market]);
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Introduce un mercado o audiencia objetivo (ej: 'startups de tecnología'). La IA analizará las tendencias visuales actuales para ese sector y te dará un informe completo.
                </p>
            </div>

            <div className="flex space-x-2">
                <input
                    type="text"
                    value={market}
                    onChange={(e) => setMarket(e.target.value)}
                    placeholder="E.g., Gen Z gamers, luxury travel, eco-friendly startups"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    disabled={isLoading}
                />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-6 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity whitespace-nowrap"
                >
                    {isLoading ? <Spinner /> : 'Analizar'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Analizando tendencias actuales...</p>}
                {result && (
                    <div className="bg-gray-800 p-6 rounded-lg space-y-6">
                        <h3 className="text-2xl font-semibold text-center">Análisis de Tendencias para: <span className="text-purple-400">{market}</span></h3>
                        
                        <div>
                            <h4 className="font-bold text-lg mb-2">Resumen General</h4>
                            <p className="text-gray-300">{result.summary}</p>
                        </div>

                         <div>
                            <h4 className="font-bold text-lg mb-2">Paletas de Colores Populares</h4>
                            <ul className="list-disc list-inside text-gray-300">{result.colorPalettes.map((c: string, i: number) => <li key={i}>{c}</li>)}</ul>
                        </div>
                        
                         <div>
                            <h4 className="font-bold text-lg mb-2">Tipografía en Tendencia</h4>
                            <ul className="list-disc list-inside text-gray-300">{result.typography.map((t: string, i: number) => <li key={i}>{t}</li>)}</ul>
                        </div>

                         <div>
                            <h4 className="font-bold text-lg mb-2">Estilo de Imaginería</h4>
                            <p className="text-gray-300">{result.imageryStyle}</p>
                        </div>
                        
                        <div>
                            <h4 className="font-bold text-lg mb-2">Conceptos Visuales</h4>
                            <div className="space-y-3">
                            {result.concepts.map((concept: any, index: number) => (
                                <div key={index} className="bg-gray-900 p-3 rounded">
                                    <p className="font-semibold text-purple-400">{concept.name}</p>
                                    <p className="text-sm text-gray-400">{concept.description}</p>
                                </div>
                            ))}
                            </div>
                        </div>

                    </div>
                )}
            </div>
        </div>
    );
};

export default TrendAnalysis;
