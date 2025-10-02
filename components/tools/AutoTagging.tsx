
import React, { useState, useCallback } from 'react';
import { generateTextWithImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface AutoTaggingProps {
    tool: Tool;
    onBack: () => void;
}

const AutoTagging: React.FC<AutoTaggingProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [tags, setTags] = useState<string[] | null>(null);

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
        setTags(null);
        try {
            const prompt = "Generate 10-15 relevant tags for this image, for a graphic designer's asset library. Tags should include objects, style, colors, and potential use cases. Return as a JSON array of strings, like [\"tag1\", \"tag2\"].";
            const response = await generateTextWithImage(prompt, imageFile, true);
            
            if(Array.isArray(response)) {
                setTags(response);
            } else if (response.rawResponse) {
                // Try to parse the raw response if it's a stringified array
                try {
                    const parsed = JSON.parse(response.rawResponse);
                    if(Array.isArray(parsed)){
                        setTags(parsed);
                    } else {
                         throw new Error("Response is not an array.");
                    }
                } catch(e) {
                     throw new Error("Could not parse the list of tags from the AI response.");
                }
            }
            else {
                throw new Error("Invalid response format from API.");
            }

        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile]);

    return (
        <div className="max-w-3xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>
            
            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube una imagen. La IA la analizará y generará una lista de etiquetas (tags) relevantes sobre su contenido, estilo y colores, facilitando la organización de tus recursos.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg text-center">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload" disabled={isLoading} />
                    <label htmlFor="image-upload" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <img src={imagePreview} alt="Preview" className="max-h-60 mx-auto rounded-lg" />
                        ) : (
                            <p>Click to upload an image to tag</p>
                        )}
                    </label>
                </div>
                 <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile}
                    className="mt-4 w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Generando Etiquetas...' : 'Generar Etiquetas'}
                </button>
            </div>


            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && <p className="text-center text-gray-400">Analizando imagen...</p>}
                {tags && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Etiquetas Sugeridas</h3>
                        <div className="flex flex-wrap justify-center gap-2">
                            {tags.map((tag, index) => (
                                <span key={index} className="bg-gray-700 text-gray-200 px-3 py-1 rounded-full text-sm">
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AutoTagging;
