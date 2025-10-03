
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { generateImage } from '../../services/geminiService';
import type { Tool } from '../../types';
import Spinner from '../ui/Spinner';

// Type assertion for SpeechRecognition
declare global {
    interface Window {
        SpeechRecognition: any;
        webkitSpeechRecognition: any;
    }
}

interface VoiceToDesignProps {
    tool: Tool;
    onBack: () => void;
}

const VoiceToDesign: React.FC<VoiceToDesignProps> = ({ tool, onBack }) => {
    const [transcript, setTranscript] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    
    const recognitionRef = useRef<any>(null);
    const speechTimeoutRef = useRef<number | null>(null);

    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setError("El reconocimiento de voz no es compatible con este navegador. Prueba con Chrome.");
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'es-ES';

        recognition.onresult = (event: any) => {
            let finalTranscript = '';
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
             setTranscript(finalTranscript + interimTranscript);

            // Debounce to detect end of speech
            if (speechTimeoutRef.current) {
                clearTimeout(speechTimeoutRef.current);
            }
            speechTimeoutRef.current = window.setTimeout(() => {
                 handleStopListening();
            }, 2000); // 2 seconds of silence
        };

        recognition.onerror = (event: any) => {
            console.error('Speech recognition error', event.error);
            setError(`Error de reconocimiento de voz: ${event.error}`);
            setIsListening(false);
        };

        recognitionRef.current = recognition;

        return () => {
            recognition.stop();
            if (speechTimeoutRef.current) clearTimeout(speechTimeoutRef.current);
        };
    }, []);

    const handleGenerate = useCallback(async (prompt: string) => {
        if (!prompt.trim()) {
            setError('La transcripción está vacía. Por favor, habla de nuevo.');
            return;
        }
        setIsLoading(true);
        setError(null);
        setGeneratedImage(null);
        try {
            const imageUrls = await generateImage(prompt, 1);
            setGeneratedImage(imageUrls[0]);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocurrió un error desconocido.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const handleStartListening = () => {
        if (recognitionRef.current && !isListening) {
            setTranscript('');
            setGeneratedImage(null);
            setError(null);
            try {
                recognitionRef.current.start();
                setIsListening(true);
            } catch (e) {
                console.error("Could not start recognition", e);
                setError("No se pudo iniciar el reconocimiento de voz. Puede que ya esté activo.");
            }
        }
    };
    
    const handleStopListening = useCallback(() => {
        if (recognitionRef.current && isListening) {
            recognitionRef.current.stop();
            setIsListening(false);
            if (transcript) {
                 handleGenerate(transcript);
            }
        }
    }, [isListening, transcript, handleGenerate]);

    return (
        <div className="max-w-4xl mx-auto">
            <button onClick={onBack} className="mb-6 text-purple-400 hover:text-purple-300">&larr; Volver al Dashboard</button>
            <div className="text-center mb-8">
                <h2 className="text-3xl font-bold">{tool.title}</h2>
                <p className="text-gray-400 mt-2">{tool.description}</p>
            </div>

            <div className="mt-6 mb-8 p-4 bg-gray-800 border border-gray-700 rounded-lg text-center">
                <p className="text-sm text-gray-300">
                    <span className="font-semibold text-purple-400">¿Cómo funciona?</span> Pulsa el botón "Empezar a Escuchar", describe tu idea en voz alta y la IA generará una imagen cuando termines de hablar.
                </p>
            </div>

            <div className="flex flex-col items-center space-y-6">
                <button
                    onClick={isListening ? handleStopListening : handleStartListening}
                    disabled={isLoading}
                    className={`relative w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 text-white font-semibold
                        ${isListening ? 'bg-red-600 animate-pulse' : 'bg-purple-600 hover:bg-purple-700'}
                        ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    <span className="absolute bottom-2 text-xs">{isListening ? 'Escuchando...' : 'Pulsar para hablar'}</span>
                </button>
                
                <div className="w-full min-h-[60px] bg-gray-800/50 p-4 rounded-lg text-center">
                    <p className="text-lg text-gray-300 italic">{transcript || "Tu transcripción aparecerá aquí..."}</p>
                </div>
            </div>

            {error && <div className="mt-6 p-4 bg-red-500/20 text-red-300 border border-red-500/30 rounded-lg">{error}</div>}
            
            <div className="mt-8">
                {isLoading && (
                    <div className="aspect-square w-full bg-gray-800 rounded-lg flex flex-col items-center justify-center animate-pulse">
                        <Spinner />
                        <p className="text-gray-500 mt-4">Generando tu idea...</p>
                    </div>
                )}
                {generatedImage && (
                    <div>
                        <h3 className="text-xl font-semibold mb-4 text-center">Resultado de tu Idea</h3>
                        <img src={generatedImage} alt="Generated from voice prompt" className="w-full aspect-square object-cover rounded-lg shadow-lg" />
                    </div>
                )}
            </div>
        </div>
    );
};

export default VoiceToDesign;
