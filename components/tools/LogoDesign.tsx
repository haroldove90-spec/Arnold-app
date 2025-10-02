
import React, { useState, useCallback } from 'react';
import { generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface LogoDesignProps {
    tool: Tool;
    onBack: () => void;
}

const LogoDesign: React.FC<LogoDesignProps> = ({ tool, onBack }) => {
    const [company, setCompany] = useState('');
    const [industry, setIndustry] = useState('');
    const [keywords, setKeywords] = useState('');
    const [style, setStyle] = useState('minimalist');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedImages, setGeneratedImages] = useState<string[]>([]);

    const handleGenerate = useCallback(async () => {
        if (!company.trim() || !industry.trim()) {
            setError('Please provide a company name and industry.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedImages([]);
        try {
            const prompt = `Generate 4 logo concepts for a company named '${company}' in the '${industry}' sector. The logo should be in a ${style} style. Incorporate the following keywords: ${keywords}. The logo should be on a clean, solid background.`;
            const imageUrls = await generateImage(prompt, 4);
            setGeneratedImages(imageUrls);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [company, industry, keywords, style]);
    
    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Rellena los campos con el nombre de la empresa, su sector, palabras clave y el estilo deseado. La IA generará 4 conceptos de logo únicos para que te sirvan de inspiración.
                </p>
            </div>

            <div className="space-y-4 bg-gray-800/50 p-6 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <input type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company Name" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                    <input type="text" value={industry} onChange={(e) => setIndustry(e.target.value)} placeholder="Industry (e.g., tech, coffee, fashion)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                </div>
                <input type="text" value={keywords} onChange={(e) => setKeywords(e.target.value)} placeholder="Keywords (e.g., modern, eco-friendly, bold)" className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading} />
                <select value={style} onChange={(e) => setStyle(e.target.value)} className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg" disabled={isLoading}>
                    <option>minimalist</option>
                    <option>vintage</option>
                    <option>futuristic</option>
                    <option>corporate</option>
                    <option>playful</option>
                </select>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Conceptos...' : 'Generar 4 Conceptos de Logo'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Generando logos... esto puede tardar un momento.</p>}
                {generatedImages.length > 0 && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Conceptos Generados</h3>
                        <div className="grid grid-cols-2 gap-4">
                            {generatedImages.map((image, index) => (
                                <img key={index} src={image} alt={`Logo concept ${index + 1}`} className="w-full aspect-square object-cover rounded-lg shadow-lg" />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default LogoDesign;
