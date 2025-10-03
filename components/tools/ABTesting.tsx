import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ABTestingProps {
    tool: Tool;
    onBack: () => void;
}

const ABTesting: React.FC<ABTestingProps> = ({ tool, onBack }) => {
    const [product, setProduct] = useState('');
    const [audience, setAudience] = useState('');
    const [goal, setGoal] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!product.trim() || !audience.trim() || !goal.trim()) {
            setError('Please fill in all fields.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `Generate 3 variations of ad copy for the following product: '${product}'. The target audience is '${audience}' and the goal is '${goal}'. For each variation, provide a headline and a body text. Also, give a brief rationale for why each variation might be effective.`;
            const schema = {
                type: 'OBJECT',
                properties: {
                    variations: {
                        type: 'ARRAY',
                        items: {
                            type: 'OBJECT',
                            properties: {
                                headline: { type: 'STRING' },
                                body: { type: 'STRING' },
                                rationale: { type: 'STRING' }
                            },
                            required: ["headline", "body", "rationale"]
                        }
                    }
                },
                required: ["variations"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [product, audience, goal]);
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe tu producto, la audiencia y el objetivo. La IA generará tres variantes de texto publicitario (título y cuerpo) para que puedas probar cuál funciona mejor.
                </p>
            </div>

            <div className="space-y-4 bg-gray-800/50 p-6 rounded-lg">
                <input type="text" value={product} onChange={(e) => setProduct(e.target.value)} placeholder="Product/Service description" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none" disabled={isLoading} />
                <input type="text" value={audience} onChange={(e) => setAudience(e.target.value)} placeholder="Target Audience (e.g., young professionals, families)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none" disabled={isLoading} />
                <input type="text" value={goal} onChange={(e) => setGoal(e.target.value)} placeholder="Campaign Goal (e.g., increase sign-ups, drive sales)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none" disabled={isLoading} />

                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Variantes...' : 'Generar Variantes de Anuncio'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Generando... por favor espere.</p>}
                {result && result.variations && (
                    <div className="space-y-6">
                        <h3 className="text-xl font-semibold mb-4 text-center">Resultados de Pruebas A/B</h3>
                        {result.variations.map((variant: any, index: number) => (
                            <div key={index} className="bg-gray-800 p-4 rounded-lg">
                                <h4 className="font-bold text-lg text-purple-400">Variante {index + 1}</h4>
                                <p className="font-semibold mt-2">{variant.headline}</p>
                                <p className="text-gray-300 mt-1">{variant.body}</p>
                                <p className="text-xs text-gray-500 mt-3 italic">Rationale: {variant.rationale}</p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ABTesting;