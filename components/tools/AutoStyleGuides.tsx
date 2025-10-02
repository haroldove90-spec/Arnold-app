
import React, { useState, useCallback } from 'react';
import { generateTextWithImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface AutoStyleGuidesProps {
    tool: Tool;
    onBack: () => void;
}

const AutoStyleGuides: React.FC<AutoStyleGuidesProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<any | null>(null);

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
            const prompt = `Generate a basic style guide based on this logo/design. Identify the primary and secondary colors (with hex codes), suggest a suitable headline and body font pairing, and describe the overall tone and style in 3-4 keywords. Respond in JSON format with the following structure: { "colors": { "primary": ["#hex1", "#hex2"], "secondary": ["#hex3", "#hex4"] }, "typography": { "headlineFont": "Font Name", "bodyFont": "Font Name", "pairingReason": "Reason..." }, "toneAndStyle": ["keyword1", "keyword2", "keyword3"] }`;
            const response = await generateTextWithImage(prompt, imageFile, true);
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
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube una imagen de tu logo o diseño principal. La IA analizará la imagen y generará una guía de estilo básica, incluyendo colores, sugerencias de tipografía y el tono de la marca.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg text-center">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                    <label htmlFor="image-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-60 mx-auto rounded-lg" />
                        ) : (
                            <p>Click to upload logo or design</p>
                        )}
                    </label>
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Guía...' : 'Generar Guía de Estilo'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Analizando diseño y generando guía...</p>}
                {result && (
                    <div className="bg-gray-800 p-6 rounded-lg">
                        <h3 className="text-2xl font-semibold mb-4 text-center">Guía de Estilo Generada</h3>
                        {result.rawResponse ? (
                           <pre className="whitespace-pre-wrap text-left bg-gray-900 p-4 rounded">{result.rawResponse}</pre>
                        ) : (
                            <div className="space-y-6">
                                <div>
                                    <h4 className="text-xl font-bold text-purple-400">Colores</h4>
                                    <div className="flex gap-4 mt-2">
                                        <div>
                                            <p className="font-semibold">Primarios:</p>
                                            <div className="flex gap-2">{result.colors?.primary?.map((c: string) => <div key={c} className="w-10 h-10 rounded" style={{backgroundColor: c}} title={c}></div>)}</div>
                                        </div>
                                        <div>
                                            <p className="font-semibold">Secundarios:</p>
                                            <div className="flex gap-2">{result.colors?.secondary?.map((c: string) => <div key={c} className="w-10 h-10 rounded" style={{backgroundColor: c}} title={c}></div>)}</div>
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <h4 className="text-xl font-bold text-purple-400">Tipografía</h4>
                                    <p><span className="font-semibold">Headline:</span> {result.typography?.headlineFont}</p>
                                    <p><span className="font-semibold">Body:</span> {result.typography?.bodyFont}</p>
                                    <p className="text-sm text-gray-400 mt-1">{result.typography?.pairingReason}</p>
                                </div>
                                <div>
                                    <h4 className="text-xl font-bold text-purple-400">Tono y Estilo</h4>
                                    <div className="flex gap-2 mt-2">
                                        {result.toneAndStyle?.map((t: string) => <span key={t} className="bg-gray-700 px-2 py-1 rounded-full text-sm">{t}</span>)}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default AutoStyleGuides;
