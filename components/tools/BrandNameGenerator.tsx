import React, { useState, useCallback } from 'react';
import { generateStructuredText } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface BrandNameGeneratorProps {
    tool: Tool;
    onBack: () => void;
}

interface BrandIdea {
    name: string;
    slogan: string;
    availability: string;
    visualConcept: string;
}

const BrandNameGenerator: React.FC<BrandNameGeneratorProps> = ({ tool, onBack }) => {
    const [description, setDescription] = useState('');
    const [keywords, setKeywords] = useState('');
    const [tone, setTone] = useState('Profesional');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<{ ideas: BrandIdea[] } | null>(null);

    const handleGenerate = useCallback(async () => {
        if (!description.trim()) {
            setError('Por favor, describe el producto o servicio.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = `Generate 3 complete brand identity concepts for a product/service. Product description: '${description}'. Keywords: '${keywords}'. The desired tone is '${tone}'. For each concept, provide a brand name, a slogan, a simulated availability check ('Probablemente Disponible' or 'Posiblemente Tomado'), and a brief visual concept idea for a logo.`;
            const schema = {
                type: 'OBJECT',
                properties: {
                    ideas: {
                        type: 'ARRAY',
                        description: "An array of 3 complete brand concepts.",
                        items: {
                            type: 'OBJECT',
                            properties: {
                                name: { type: 'STRING' },
                                slogan: { type: 'STRING' },
                                availability: { type: 'STRING', description: "Simulated availability check." },
                                visualConcept: { type: 'STRING', description: "A brief logo/visual idea." }
                            },
                            required: ["name", "slogan", "availability", "visualConcept"]
                        }
                    }
                },
                required: ["ideas"]
            };
            const response = await generateStructuredText(prompt, schema);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [description, keywords, tone]);
    
    const AvailabilityBadge: React.FC<{ status: string }> = ({ status }) => {
        const isAvailable = status === 'Probablemente Disponible';
        return (
            <span className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${isAvailable ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                {status}
            </span>
        );
    };

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Describe tu producto, añade palabras clave y elige un tono. La IA generará conceptos de marca completos, incluyendo nombre, slogan, disponibilidad y una idea para el logo.
                </p>
            </div>

            <div className="space-y-4 bg-gray-800/50 p-6 rounded-lg">
                <textarea
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Descripción del Producto/Servicio (ej: Una app de café artesanal a domicilio)"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={3}
                    disabled={isLoading}
                />
                 <input
                    type="text"
                    value={keywords}
                    onChange={(e) => setKeywords(e.target.value)}
                    placeholder="Palabras Clave (ej: rápido, fresco, premium)"
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    disabled={isLoading}
                />
                 <select
                    value={tone}
                    onChange={(e) => setTone(e.target.value)}
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    disabled={isLoading}
                >
                    <option>Profesional</option>
                    <option>Divertido</option>
                    <option>Moderno</option>
                    <option>Lujoso</option>
                    <option>Minimalista</option>
                </select>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Ideas...' : 'Generar Conceptos de Marca'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Buscando inspiración...</p>}
                {result && result.ideas && (
                    <div className="space-y-6">
                        {result.ideas.map((idea, index) => (
                           <div key={index} className="bg-gray-800 border border-gray-700/50 p-6 rounded-lg transform hover:scale-[1.02] transition-transform duration-300">
                                <div className="flex justify-between items-start mb-2">
                                    <h3 className="text-2xl font-bold text-purple-400">{idea.name}</h3>
                                    <AvailabilityBadge status={idea.availability} />
                                </div>
                                <p className="text-lg italic text-gray-300 mb-4">"{idea.slogan}"</p>
                                <div>
                                    <h4 className="font-semibold text-gray-200">Concepto Visual:</h4>
                                    <p className="text-gray-400 text-sm">{idea.visualConcept}</p>
                                </div>
                           </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default BrandNameGenerator;