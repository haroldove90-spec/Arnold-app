import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface WebUIDesignProps {
    tool: Tool;
    onBack: () => void;
}

const WebUIDesign: React.FC<WebUIDesignProps> = ({ tool, onBack }) => {
    const [pageType, setPageType] = useState('');
    const [business, setBusiness] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!pageType.trim() || !business.trim()) {
            setError('Please describe the page type and business.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `Provide a layout suggestion for a '${pageType}' webpage for a '${business}'. Describe the key sections (e.g., Hero, Features, Testimonials, CTA) and the components within each. Suggest a visual style. Do not generate code, but a structured description.`;
            const schema = {
                type: 'OBJECT',
                properties: {
                    visualStyle: { type: 'STRING' },
                    sections: {
                        type: 'ARRAY',
                        items: {
                            type: 'OBJECT',
                            properties: {
                                name: { type: 'STRING' },
                                description: { type: 'STRING' },
                                components: { type: 'ARRAY', items: { type: 'STRING' } }
                            },
                             required: ["name", "description", "components"]
                        }
                    }
                },
                required: ["visualStyle", "sections"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [pageType, business]);
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe el tipo de página web que necesitas y para qué tipo de negocio es. La IA te sugerirá una estructura de página (layout) con las secciones y componentes clave.
                </p>
            </div>

            <div className="space-y-4 bg-gray-800/50 p-6 rounded-lg">
                <input type="text" value={pageType} onChange={(e) => setPageType(e.target.value)} placeholder="Page Type (e.g., landing page, portfolio, blog)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                <input type="text" value={business} onChange={(e) => setBusiness(e.target.value)} placeholder="Business / Project Type (e.g., SaaS startup, photographer, recipe site)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Sugerencias...' : 'Sugerir Layout'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Diseñando la estructura de tu web...</p>}
                {result && (
                    <div className="bg-gray-800 p-6 rounded-lg space-y-6">
                        <h3 className="text-2xl font-semibold text-center">Sugerencia de Layout para: <span className="text-purple-400">{pageType}</span></h3>
                        <p className="text-center"><span className="font-semibold">Estilo Visual Sugerido:</span> {result.visualStyle}</p>
                        <div className="space-y-4">
                            {result.sections.map((section: any, index: number) => (
                                <div key={index} className="bg-gray-900 p-4 rounded">
                                    <h4 className="font-bold text-lg text-purple-400">{section.name}</h4>
                                    <p className="text-sm text-gray-300 mt-1">{section.description}</p>
                                    <div className="mt-2">
                                        <p className="text-xs font-semibold">Componentes:</p>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {section.components.map((comp: string, i: number) => (
                                                <span key={i} className="bg-gray-700 text-xs px-2 py-1 rounded-full">{comp}</span>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default WebUIDesign;