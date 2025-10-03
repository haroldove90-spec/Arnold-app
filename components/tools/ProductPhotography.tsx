
import React, { useState, useCallback } from 'react';
import { editImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

interface ProductPhotographyProps {
    tool: Tool;
    onBack: () => void;
}

const ProductPhotography: React.FC<ProductPhotographyProps> = ({ tool, onBack }) => {
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [resultImage, setResultImage] = useState<string | null>(null);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            setResultImage(null);
            setImagePreview(URL.createObjectURL(file));
        }
    };

    const handleGenerate = useCallback(async () => {
        if (!imageFile) {
            setError('Por favor, sube una imagen de producto.');
            return;
        }
        if (!prompt.trim()) {
            setError('Por favor, describe la mejora que deseas.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setResultImage(null);
        try {
            const response = await editImage(prompt, imageFile);
            setResultImage(response.image);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [imageFile, prompt]);

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>
            
            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Sube la foto de un producto y describe cómo quieres mejorarla. Puedes pedir un nuevo fondo, un cambio de iluminación, o un estilo diferente.
                </p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg space-y-4">
                <div className="w-full p-4 border-2 border-dashed border-gray-600 rounded-lg text-center">
                    <input type="file" accept="image/*" onChange={handleImageChange} className="hidden" id="image-upload-product" disabled={isLoading} />
                    <label htmlFor="image-upload-product" className={`cursor-pointer ${isLoading ? 'cursor-not-allowed' : ''}`}>
                        {imagePreview ? (
                            <p>Imagen de producto seleccionada.</p>
                        ) : (
                            <p>Haz clic para subir una foto de producto</p>
                        )}
                    </label>
                </div>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="Ej: Colocar sobre una mesa de mármol con una planta borrosa al fondo."
                    className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
                    rows={2}
                    disabled={isLoading}
                />
                 <button
                    onClick={handleGenerate}
                    disabled={isLoading || !imageFile || !prompt.trim()}
                    className="w-full flex justify-center items-center bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold py-3 px-4 rounded-lg hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                >
                    {isLoading && <Spinner />}
                    {isLoading ? 'Mejorando Foto...' : 'Mejorar Foto de Producto'}
                </button>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Original</h3>
                    <div className="w-full aspect-square bg-gray-700 rounded-lg p-2 flex items-center justify-center">
                        {imagePreview ? <img src={imagePreview} alt="Original product" className="max-w-full max-h-full object-contain" /> : <p className="text-gray-400">Sube una imagen</p>}
                    </div>
                </div>
                 <div>
                    <h3 className="text-xl font-semibold mb-4 text-center">Resultado</h3>
                     <div className="w-full aspect-square bg-gray-700 rounded-lg shadow-lg p-2">
                        {isLoading && <div className="w-full h-full flex items-center justify-center animate-pulse bg-gray-800/50 rounded-md"><p>Procesando...</p></div>}
                        {resultImage && <img src={resultImage} alt="Enhanced product" className="w-full h-full object-contain" />}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ProductPhotography;
