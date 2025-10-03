import { Type } from "@google/genai";

// Convierte un objeto File a un formato serializable (Base64) para enviarlo en el cuerpo de una petición JSON.
async function fileToSerializable(file: File): Promise<{ data: string; mimeType: string }> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            if (typeof reader.result !== 'string') {
                return reject(new Error("Failed to read file as data URL."));
            }
            const base64String = reader.result.split(',')[1];
            resolve({ data: base64String, mimeType: file.type });
        };
        reader.onerror = (error) => reject(error);
        reader.readAsDataURL(file);
    });
}

// Un manejador centralizado para llamar a nuestra API serverless de forma segura.
async function callApi<T>(action: string, payload: object): Promise<T> {
    try {
        const response = await fetch('/api/gemini', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, payload }),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'An unknown API error occurred.');
        }

        return result.data;
    } catch (error) {
        console.error(`API call failed for action "${action}":`, error);
        if (error instanceof Error && error.message.includes("API_KEY is not configured")) {
            throw new Error("Error de Configuración: La API Key para el servicio de IA no ha sido configurada. El administrador del sitio debe añadirla en los ajustes de despliegue para que la aplicación funcione.");
        }
        throw error;
    }
}

// Las funciones de servicio ahora son simples y limpias, delegando la lógica a nuestro manejador de API.

export const generateImage = (prompt: string, numberOfImages: number = 1): Promise<string[]> => {
    return callApi('generateImage', { prompt, numberOfImages });
};

export const generateStructuredText = (prompt: string, responseSchema: any): Promise<any> => {
    return callApi('generateStructuredText', { prompt, responseSchema });
};

export const generateText = (prompt: string): Promise<string> => {
    return callApi('generateText', { prompt });
};

export const generateTextWithImage = async (prompt: string, image: File, isJson: boolean = false): Promise<any> => {
    const serializableImage = await fileToSerializable(image);
    return callApi('generateTextWithImage', { prompt, image: serializableImage, isJson });
};

export const editImage = async (prompt: string, image: File): Promise<{ text: string | null; image: string | null }> => {
    const serializableImage = await fileToSerializable(image);
    return callApi('editImage', { prompt, image: serializableImage });
};

// --- Tipos y funciones específicas que usan los servicios anteriores ---

export interface Color {
    name: string;
    hex: string;
}

export const generateColorPalette = async (theme: string): Promise<Color[]> => {
    const prompt = `Generate a color palette with 5 colors for the theme: ${theme}. Provide creative names for each color.`;
    const schema = {
        type: Type.OBJECT,
        properties: {
            palette: {
                type: Type.ARRAY,
                description: 'An array of 5 color objects.',
                items: {
                    type: Type.OBJECT,
                    properties: {
                        name: { type: Type.STRING, description: 'The creative name of the color.' },
                        hex: { type: Type.STRING, description: 'The hex code for the color (e.g., #RRGGBB).' },
                    },
                    required: ["name", "hex"]
                },
            },
        },
        required: ["palette"]
    };
    
    try {
        const result = await generateStructuredText(prompt, schema);
        if (result.palette && Array.isArray(result.palette)) {
            return result.palette;
        } else {
            throw new Error("Invalid response format from API for color palette.");
        }
    } catch (error) {
         if (error instanceof Error) {
            throw error; // Re-lanza el error ya procesado por callApi
         }
         throw new Error("An unknown error occurred while generating the color palette.");
    }
};
