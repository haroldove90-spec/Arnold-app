import { GoogleGenAI, Type, Modality } from "@google/genai";

let aiInstance: GoogleGenAI | null = null;

/**
 * Lazily initializes and returns the GoogleGenAI instance.
 * Throws a user-friendly error if the API key is not configured.
 */
const getAiInstance = (): GoogleGenAI => {
    if (!aiInstance) {
        const apiKey = process.env.API_KEY;
        // This check handles null, undefined, and empty strings, preventing the API call.
        if (!apiKey) {
             throw new Error("Error de Configuración: La API Key para el servicio de IA no ha sido configurada. El administrador del sitio debe añadirla en los ajustes de despliegue para que la aplicación funcione.");
        }
        aiInstance = new GoogleGenAI({ apiKey });
    }
    return aiInstance;
};

const handleError = (error: unknown, context: string): never => {
    console.error(`Error in ${context}:`, error);

    if (error instanceof Error) {
        // If it's our specific, self-thrown configuration error, re-throw it as is.
        if (error.message.startsWith("Error de Configuración")) {
            throw error;
        }

        // As a fallback, try to detect the API key error from the backend response.
        let isApiKeyError = false;
        if (error.message.includes("API key not valid") || error.message.includes("API_KEY_INVALID")) {
            isApiKeyError = true;
        } else {
            // Try to parse the message as JSON for a more robust check.
            try {
                const parsedError = JSON.parse(error.message);
                const details = parsedError?.error?.details;
                if (details && Array.isArray(details)) {
                     if (details.some((d: any) => d.reason === 'API_KEY_INVALID')) {
                        isApiKeyError = true;
                     }
                }
            } catch (e) {
                // Not a JSON string.
            }
        }
        
        if (isApiKeyError) {
             throw new Error("Error de Configuración: La API Key para el servicio de IA no ha sido configurada. El administrador del sitio debe añadirla en los ajustes de despliegue para que la aplicación funcione.");
        }

        throw new Error(`Failed to ${context}. ${error.message}`);
    }
    
    throw new Error(`An unknown error occurred while trying to ${context}.`);
};


async function fileToGenerativePart(file: File) {
    const base64EncodedDataPromise = new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve((reader.result as string).split(',')[1]);
        reader.readAsDataURL(file);
    });
    return {
        inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
    };
}

export const generateImage = async (prompt: string, numberOfImages: number = 1): Promise<string[]> => {
    try {
        const ai = getAiInstance();
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: prompt,
            config: {
                numberOfImages,
                outputMimeType: 'image/jpeg',
                aspectRatio: '1:1',
            },
        });

        if (response.generatedImages && response.generatedImages.length > 0) {
            return response.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
        } else {
            throw new Error("No images were generated.");
        }
    } catch (error) {
        handleError(error, "generate image");
    }
};

export const generateStructuredText = async (prompt: string, responseSchema: any): Promise<any> => {
    try {
        const ai = getAiInstance();
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema,
            },
        });
        
        const jsonString = response.text.trim();
        return JSON.parse(jsonString);
    } catch (error) {
        handleError(error, "generate structured text");
    }
};

export const generateText = async (prompt: string): Promise<string> => {
    try {
        const ai = getAiInstance();
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });
        return response.text;
    } catch (error) {
        handleError(error, "generate text");
    }
};


export const generateTextWithImage = async (prompt: string, image: File, isJson: boolean = false): Promise<any> => {
    try {
        const ai = getAiInstance();
        const imagePart = await fileToGenerativePart(image);
        const textPart = { text: prompt };
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: { parts: [imagePart, textPart] },
            ...(isJson && { 
                config: {
                    responseMimeType: "application/json" 
                }
            })
        });
        
        const textResponse = response.text;
        if (isJson) {
            try {
                return JSON.parse(textResponse);
            } catch (e) {
                // The API might not respect the JSON format if the input is too ambiguous.
                // We return the raw text for the user to see.
                return { rawResponse: textResponse };
            }
        }
        return textResponse;

    } catch (error) {
        handleError(error, "generate text with image");
    }
};

export const editImage = async (prompt: string, image: File): Promise<{text: string | null, image: string | null}> => {
    try {
        const ai = getAiInstance();
        const imagePart = await fileToGenerativePart(image);
        const textPart = { text: prompt };
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: [imagePart, textPart] },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });
        
        let resultText: string | null = null;
        let resultImage: string | null = null;

        for (const part of response.candidates[0].content.parts) {
            if (part.text) {
                resultText = part.text;
            } else if (part.inlineData) {
                const base64ImageBytes: string = part.inlineData.data;
                const mimeType = part.inlineData.mimeType;
                resultImage = `data:${mimeType};base64,${base64ImageBytes}`;
            }
        }
        
        if (!resultImage) {
            throw new Error("The AI did not return an edited image.");
        }

        return { text: resultText, image: resultImage };

    } catch (error) {
        handleError(error, "edit image");
    }
};


export interface Color {
    name: string;
    hex: string;
}

export const generateColorPalette = async (theme: string): Promise<Color[]> => {
    try {
        const prompt = `Generate a color palette with 5 colors for the theme: ${theme}. Provide creative names for each color.`;
        const result = await generateStructuredText(prompt, {
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
        });

        if (result.palette && Array.isArray(result.palette)) {
            return result.palette;
        } else {
            throw new Error("Invalid response format from API.");
        }
    } catch (error) {
       handleError(error, "generate color palette");
    }
};