import React, { useState, useCallback } from 'react';
import { analyzeOriginality } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface PlagiarismDetectionProps {
    tool: Tool;
    onBack: () => void;
}

// Estructura para el resultado del análisis
interface Similarity {
    title: string;
    description: string;
}

interface AnalysisResult {
    originalityScore: number; // Puntuación de 1 a 10
    summary: string;
    similarDesigns: Similarity[];
    similarImages?: string[]; // Array de URLs de imágenes base64
    rawResponse?: string; // Fallback por si el JSON falla
}

const PlagiarismDetection: React.FC<PlagiarismDetectionProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
            setAnalysis(null);
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Por favor, sube una imagen.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setAnalysis(null);
        try {
            const prompt = `Act as a design originality expert. Analyze this image and search for visually similar designs, concepts, or logos. Provide a detailed report on its originality. Respond in JSON format with the following structure: { "originalityScore": <a number between 1 and 10 where 10 is completely unique>, "summary": "<a brief analysis summary>", "similarDesigns": [{ "title": "<name of similar design/concept>", "description": "<brief description of the similar design and why it's relevant>" }] }. If you find no similar designs, the similarDesigns array should be empty.`;
            const response = await analyzeOriginality(prompt, imageFile);
            setAnalysis(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);

    const getScoreColor = (score: number) => {
        if (score >= 8) return 'text-green-400';
        if (score >= 5) return 'text-yellow-400';
        return 'text-red-400';
    }

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube la imagen de tu diseño. La IA buscará en la web para encontrar diseños visualmente similares y te dará un informe sobre la originalidad, ¡incluyendo ejemplos visuales!
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                    <label htmlFor="image-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-60 mx-auto rounded-lg" />
                        ) : (
                            <p>Haz clic para subir un diseño y verificar su originalidad</p>
                        )}
                    </label>
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Verificando...' : 'Verificar Originalidad'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">La IA está buscando en la web diseños similares...</p>}
                {analysis && (
                    analysis.rawResponse ? (
                        <div>
                            <h3 className="text-xl font-semibold mb-4 text-center">Análisis de Originalidad (Fallback)</h3>
                             <div className="bg-gray-800 p-6 rounded-lg">
                                <p className="text-yellow-400 text-sm mb-4">La IA no devolvió un formato estructurado. Mostrando respuesta en bruto:</p>
                                <pre className="whitespace-pre-wrap text-left bg-gray-900 p-4 rounded font-sans text-gray-300">{analysis.rawResponse}</pre>
                            </div>
                        </div>
                    ) : (
                    <div className="bg-gray-800 p-6 rounded-lg space-y-6">
                        <h3 className="text-2xl font-semibold text-center text-white">Reporte de Originalidad</h3>
                        
                        <div className="text-center">
                            <h4 className="text-lg font-bold text-purple-400">Puntuación de Originalidad</h4>
                            <p className={`text-6xl font-bold ${getScoreColor(analysis.originalityScore)}`}>{analysis.originalityScore}<span className="text-3xl text-gray-500">/10</span></p>
                        </div>

                        <div>
                            <h4 className="text-lg font-bold text-purple-400 mb-2">Resumen del Análisis</h4>
                            <p className="text-gray-300 bg-gray-900/50 p-4 rounded-md">{analysis.summary}</p>
                        </div>
                        
                        {analysis.similarDesigns && analysis.similarDesigns.length > 0 && (
                            <div>
                                <h4 className="text-lg font-bold text-purple-400 mb-2">Posibles Similitudes Encontradas</h4>
                                <p className="text-sm text-gray-400 mb-4">La IA ha identificado los siguientes conceptos como potencialmente similares y ha generado una visualización de ellos.</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {analysis.similarDesigns.slice(0, 2).map((design, index) => (
                                    <div key={index} className="bg-gray-900 p-4 rounded-lg border-l-4 border-gray-700">
                                        {analysis.similarImages && analysis.similarImages[index] && (
                                            <img src={analysis.similarImages[index]} alt={design.title} className="w-full aspect-square object-cover rounded-md mb-3" />
                                        )}
                                        <p className="font-semibold text-white">{design.title || `Diseño Similar ${index + 1}`}</p>
                                        <p className="text-sm text-gray-400 mt-1">{design.description}</p>
                                    </div>
                                ))}
                                </div>
                            </div>
                        )}
                         {analysis.similarDesigns && analysis.similarDesigns.length === 0 && (
                             <div>
                                <h4 className="text-lg font-bold text-purple-400 mb-2">Posibles Similitudes Encontradas</h4>
                                <p className="text-gray-300 bg-gray-900/50 p-4 rounded-md">No se encontraron similitudes visuales significativas. El diseño parece ser original.</p>
                            </div>
                         )}
                    </div>
                    )
                )}
            </div>
        </div>
    );
};

export default PlagiarismDetection;