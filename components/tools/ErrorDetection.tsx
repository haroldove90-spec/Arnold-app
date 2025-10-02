
import React, { useState, useCallback } from 'react';
import { generateTextWithImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ErrorDetectionProps {
    tool: Tool;
    onBack: () => void;
}

const ErrorDetection: React.FC<ErrorDetectionProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Please upload an image.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResult(null);
        try {
            const prompt = "Analyze this image for potential technical design errors. Check for things like low resolution (if apparent), potential color space issues for print (e.g., using pure RGB black instead of a rich black), and typographic mistakes like kerning, tracking, or readability problems. Provide a list of potential issues found.";
            const response = await generateTextWithImage(prompt, imageFile);
            setResult(response);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube tu diseño final. La IA lo revisará en busca de errores técnicos comunes, como baja resolución, problemas de color para impresión o errores tipográficos.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                    <label htmlFor="image-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-60 mx-auto rounded-lg" />
                        ) : (
                            <p>Click to upload a design to check for errors</p>
                        )}
                    </label>
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Analizando Diseño...' : 'Detectar Errores Potenciales'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">La IA está buscando errores en tu diseño...</p>}
                {result && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Análisis de Errores Potenciales</h3>
                        <div className="bg-gray-800 p-6 rounded-lg">
                           <pre className="whitespace-pre-wrap text-left bg-gray-900 p-4 rounded font-sans text-gray-300">{result}</pre>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ErrorDetection;
