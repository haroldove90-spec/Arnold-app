import type { VercelRequest, VercelResponse } from '@vercel/node';
import { GoogleGenAI, Modality } from "@google/genai";

const getAiInstance = () => {
    const apiKey = process.env.API_KEY;
    if (!apiKey) {
        throw new Error("API_KEY is not configured in Vercel environment variables.");
    }
    return new GoogleGenAI({ apiKey });
};

export default async function handler(req: VercelRequest, res: VercelResponse) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

    try {
        const ai = getAiInstance();
        const { action, payload } = req.body;

        switch (action) {
            case 'generateImage': {
                const { prompt, numberOfImages } = payload;
                const response = await ai.models.generateImages({
                    model: 'imagen-4.0-generate-001',
                    prompt,
                    config: { numberOfImages, outputMimeType: 'image/jpeg', aspectRatio: '1:1' },
                });
                const imageUrls = response.generatedImages.map(img => `data:image/jpeg;base64,${img.image.imageBytes}`);
                return res.status(200).json({ data: imageUrls });
            }

            case 'generateStructuredText': {
                 const { prompt, responseSchema } = payload;
                 const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: prompt,
                    config: { responseMimeType: "application/json", responseSchema },
                 });
                 return res.status(200).json({ data: JSON.parse(response.text.trim()) });
            }

            case 'generateText': {
                const { prompt } = payload;
                const response = await ai.models.generateContent({ model: "gemini-2.5-flash", contents: prompt });
                return res.status(200).json({ data: response.text });
            }

            case 'generateTextWithImage': {
                const { prompt, image, isJson } = payload;
                const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
                const textPart = { text: prompt };
                const response = await ai.models.generateContent({
                    model: "gemini-2.5-flash",
                    contents: { parts: [imagePart, textPart] },
                    ...(isJson && { config: { responseMimeType: "application/json" } })
                });
                const textResponse = response.text;
                if (isJson) {
                    try {
                        return res.status(200).json({ data: JSON.parse(textResponse) });
                    } catch (e) {
                        return res.status(200).json({ data: { rawResponse: textResponse } });
                    }
                }
                return res.status(200).json({ data: textResponse });
            }

            case 'editImage': {
                const { prompt, image } = payload;
                const imagePart = { inlineData: { data: image.data, mimeType: image.mimeType } };
                const textPart = { text: prompt };
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: { parts: [imagePart, textPart] },
                    config: { responseModalities: [Modality.IMAGE, Modality.TEXT] },
                });

                let resultText: string | null = null;
                let resultImage: string | null = null;

                for (const part of response.candidates[0].content.parts) {
                    if (part.text) resultText = part.text;
                    else if (part.inlineData) {
                        resultImage = `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
                    }
                }
                
                if (!resultImage) {
                   throw new Error("The AI did not return an edited image.");
                }
                
                return res.status(200).json({ data: { text: resultText, image: resultImage } });
            }

            default:
                return res.status(400).json({ error: 'Invalid action' });
        }
    } catch (error: unknown) {
        console.error(`Error in API handler:`, error);
        const errorMessage = error instanceof Error ? error.message : 'An unknown server error occurred.';
        return res.status(500).json({ error: errorMessage });
    }
}
