import { NextRequest, NextResponse } from "next/server";
import { db, uploadedFiles } from "@/drizzle/schema";
import { eq } from "drizzle-orm";
import { del, list } from "@vercel/blob";
import { Mistral } from "@mistralai/mistralai";
import { incrementAndLogTokenUsage } from "@/lib/incrementAndLogTokenUsage";
import { auth } from "@clerk/nextjs/server";
import sharp from 'sharp';
import { FilePurpose } from "@mistralai/mistralai/models/components";

export const maxDuration = 60; // 5 minutes

// Define interfaces for OCR responses based on Mistral's API
interface PageContent {
  text?: string;
  content?: string;
  markdown?: string;
  [key: string]: unknown; // Use unknown instead of any
}

interface MistralOCRResponse {
  pages: Array<string | PageContent>;
  usageInfo?: {
    totalTokens?: number;
    promptTokens?: number;
    completionTokens?: number;
    total_tokens?: number; // Fallback for older API versions
  };
  text?: string;
  content?: string;
  markdown?: string;
  [key: string]: unknown; // Use unknown instead of any
}

// Define extended FilePurpose type to include "ocr" which might not be in the SDK yet
type MistralFilePurpose = 
  | "fine-tuning"
  | "assistants"
  | "batch"
  | "ocr" // Add OCR purpose which might be missing from SDK types
  | string; // Fallback to allow for future purposes without breaking

// Define custom error type
interface ProcessingError extends Error {
  message: string;
  stack?: string;
  code?: string;
  statusCode?: number;
}

// Helper function to compress image
async function compressImage(buffer: Buffer, fileType: string): Promise<Buffer> {
  // Skip compression for non-image files
  if (!fileType || 
      fileType.includes('pdf') || 
      fileType === 'application/pdf' || 
      !fileType.startsWith('image/')) {
    return buffer; // Return original buffer for non-image files
  }
  
  try {
    // Process image with Sharp
    return await sharp(buffer)
      .resize(800, 800, { // Resize to max 800x800 while maintaining aspect ratio
        fit: 'inside',
        withoutEnlargement: true
      })
      .jpeg({ // Convert to JPEG with moderate compression
        quality: 60,
        mozjpeg: true
      })
      .toBuffer();
  } catch (error) {
    console.error('Image compression error:', error);
    return buffer; // Return original buffer if compression fails
  }
}

export async function POST(request: NextRequest) {
  let fileId: number | null = null;
  
  try {
    // Check authentication first
    const { userId } = await auth();
    const authHeader = request.headers.get("authorization");
    const payload = await request.json() as { fileId: number };
    fileId = payload.fileId;
    
    // Handle API key auth from mobile app
    if (!userId && authHeader) {
      const token = authHeader.replace("Bearer ", "");
      
      if (!token) {
        console.error("Unauthorized process attempt - invalid token");
        return NextResponse.json(
          { error: "Unauthorized" },
          { status: 401 }
        );
      }
      
      // For mobile requests with a token, we need to validate the token
      // Extract userId from token for mobile auth
      let tokenUserId = null;
      
      try {
        // Log token for debugging
        console.log("Processing mobile token:", token.substring(0, 20) + "...");
        
        // Basic JWT structure check
        const parts = token.split('.');
        if (parts.length === 3) {
          // This is likely a JWT - try to decode the payload
          // Add proper base64 padding if needed
          let base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
          while (base64.length % 4) {
            base64 += '=';
          }
          
          const payload = JSON.parse(Buffer.from(base64, 'base64').toString()) as {
            sub?: string;
            userId?: string;
            user_id?: string;
            [key: string]: unknown; // Use unknown instead of any
          };
          console.log("Decoded token payload:", JSON.stringify(payload));
          
          // Check for user ID in different possible claims
          if (payload.sub) {
            tokenUserId = payload.sub;
            console.log("Extracted userId from sub claim:", tokenUserId);
          } else if (payload.userId) {
            tokenUserId = payload.userId;
            console.log("Extracted userId from userId claim:", tokenUserId);
          } else if (payload.user_id) {
            tokenUserId = payload.user_id;
            console.log("Extracted userId from user_id claim:", tokenUserId);
          }
        }
      } catch (parseError) {
        console.error("Error parsing token:", parseError);
        return NextResponse.json(
          { error: "Invalid token format" },
          { status: 401 }
        );
      }
      
      // Continue with file processing for mobile
      if (!fileId) {
        return NextResponse.json(
          { error: "File ID is required" },
          { status: 400 }
        );
      }
      
      // Get the file record
      const [file] = await db
        .select()
        .from(uploadedFiles)
        .where(eq(uploadedFiles.id, fileId))
        .limit(1);
      
      if (!file) {
        return NextResponse.json(
          { error: "File not found" },
          { status: 404 }
        );
      }
      
      // For debugging only - log user IDs but don't reject yet
      if (tokenUserId && file.userId !== tokenUserId) {
        console.log(`Notice: File userId (${file.userId}) doesn't match token userId (${tokenUserId}), but allowing for testing`);
        // We'll temporarily allow access even if the user IDs don't match
        // return NextResponse.json(
        //   { error: "Unauthorized" },
        //   { status: 401 }
        // );
      }
      
      // Update the file status to processing
      await db
        .update(uploadedFiles)
        .set({ status: "processing" })
        .where(eq(uploadedFiles.id, fileId));
      
      // Return success response
      return NextResponse.json({ success: true });
    } else if (!userId) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    // Regular web authentication flow
    if (!fileId) {
      return NextResponse.json(
        { error: "File ID is required" },
        { status: 400 }
      );
    }

    // Get file record
    const [file] = await db
      .select()
      .from(uploadedFiles)
      .where(eq(uploadedFiles.id, fileId))
      .limit(1);

    if (!file) {
      return NextResponse.json(
        { error: "File not found" },
        { status: 404 }
      );
    }

    if (file.userId !== userId) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    // Update status to processing
    await db
      .update(uploadedFiles)
      .set({ status: "processing" })
      .where(eq(uploadedFiles.id, fileId));

    // Get file from Blob storage
    const response = await fetch(file.blobUrl);
    if (!response.ok) {
      throw new Error("Failed to fetch file from storage");
    }
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    
    // Initialize Mistral client
    const apiKey = process.env.MISTRAL_API_KEY;
    if (!apiKey) {
      throw new Error("Mistral API key is not configured");
    }
    const mistralClient = new Mistral({ apiKey });
    
    // Determine file type and handling approach
    const fileType = file.fileType.toLowerCase();
    let textContent: string = '';
    let tokensUsed: number = 0;
    
    try {
      if (fileType === 'application/pdf' || fileType.includes('pdf')) {
        // For PDFs, use the document OCR processing
        // 1. Upload file to Mistral
        const uploadedFile = await mistralClient.files.upload({
          file: {
            fileName: file.originalName || `file-${fileId}.pdf`,
            // @ts-ignore
            content: buffer,
          },
          // Use our custom type instead of 'any'
          purpose: "ocr" as FilePurpose
        });
        
        console.log("Uploaded PDF file to Mistral:", uploadedFile.id);
        
        // 2. Get signed URL for the uploaded file
        const signedUrl = await mistralClient.files.getSignedUrl({
          fileId: uploadedFile.id,
        });
        
        console.log("Got signed URL for PDF:", signedUrl.url.substring(0, 50) + "...");
        
        // 3. Process the PDF with OCR
        const ocrResponse = await mistralClient.ocr.process({
          model: "mistral-ocr-latest",
          document: {
            type: "document_url",
            documentUrl: signedUrl.url,
          }
        }) as MistralOCRResponse;
        
        console.log("OCR processing complete, extracting text content");
        console.log("OCR response structure:", JSON.stringify(ocrResponse, null, 2).substring(0, 500) + "...");
        
        // 4. Extract text from OCR response with improved handling
        if (ocrResponse) {
          // Try to extract from different possible response structures
          if (ocrResponse.pages && Array.isArray(ocrResponse.pages)) {
            console.log(`Found ${ocrResponse.pages.length} pages in OCR response`);
            
            textContent = ocrResponse.pages.map((page, index) => {
              console.log(`Processing page ${index + 1}, type: ${typeof page}`);
              
              if (typeof page === 'string') {
                return page;
              } else if (typeof page === 'object' && page !== null) {
                // Log available properties on this page object
                console.log(`Page ${index + 1} properties:`, Object.keys(page));
                
                // Check for different property names that might contain text
                if (page.text) return page.text;
                if (page.content) return page.content;
                if (page.markdown) return page.markdown;
                
                // Try to stringify the entire object if we couldn't find specific text properties
                try {
                  return JSON.stringify(page);
                } catch (e) {
                  console.error(`Error stringifying page ${index + 1}:`, e);
                  return '';
                }
              }
              return '';
            }).join("\n\n");
            
            console.log(`Extracted ${textContent.length} characters of text content`);
            
            // Fallback if text is still empty
            if (!textContent.trim()) {
              console.log("No text extracted from pages array. Trying alternate approaches.");
              
              // Check if there's a simpler text property directly on the response
              if (ocrResponse.text) {
                textContent = ocrResponse.text;
                console.log("Found text directly on OCR response");
              } else if (typeof ocrResponse === 'string') {
                textContent = ocrResponse;
                console.log("OCR response is a string, using directly");
              } else {
                // Last resort: stringify the response object, but clean up to extract only text
                try {
                  const fullResponseString = JSON.stringify(ocrResponse);
                  console.log("Using full response string as fallback");
                  textContent = fullResponseString;
                } catch (e) {
                  console.error("Error converting OCR response to string:", e);
                }
              }
            }
          } else {
            console.log("OCR response does not have a pages array, structure:", 
                        typeof ocrResponse, 
                        Object.keys(ocrResponse));
            
            // Try to extract text from other potential response formats
            if (ocrResponse.text) {
              textContent = ocrResponse.text;
            } else if (ocrResponse.content) {
              textContent = ocrResponse.content;
            } else if (ocrResponse.markdown) {
              textContent = ocrResponse.markdown;
            } else {
              // Try to stringify response for inspection
              try {
                textContent = JSON.stringify(ocrResponse);
                console.log("Using stringified response:", textContent.substring(0, 100) + "...");
              } catch (e) {
                console.error("Error stringifying OCR response:", e);
                textContent = "Error extracting text from OCR response";
              }
            }
          }
        } else {
          console.error("OCR response is null or undefined");
          textContent = "Error: Could not obtain OCR response";
        }
        
        // 5. Extract token usage information
        if (ocrResponse?.usageInfo) {
          if (ocrResponse.usageInfo.totalTokens) {
            tokensUsed = ocrResponse.usageInfo.totalTokens;
          } else if (ocrResponse.usageInfo.total_tokens) {
            tokensUsed = ocrResponse.usageInfo.total_tokens;
          }
          console.log("Token usage:", tokensUsed);
        } else {
          console.log("No usage info available in OCR response");
        }
        
        // Optional: Delete the uploaded file to clean up
        // Commented out for now to prevent potential issues
        /*
        try {
          await mistralClient.files.delete({
            fileId: uploadedFile.id,
          });
          console.log("Deleted uploaded file from Mistral");
        } catch (deleteError) {
          console.error("Failed to delete uploaded file:", deleteError);
          // Continue processing even if deletion fails
        }
        */
      } else {
        // For images, compress first if needed
        const processedBuffer = await compressImage(buffer, fileType);
        
        // 1. Upload image to Mistral
        const uploadedImage = await mistralClient.files.upload({
          file: {
            fileName: file.originalName || `image-${fileId}.jpg`,
            // @ts-ignore
            content: processedBuffer,
          },
          purpose: "ocr" as FilePurpose
        });
        
        console.log("Uploaded image file to Mistral:", uploadedImage.id);
        
        // 2. Get signed URL for the uploaded image
        const signedUrl = await mistralClient.files.getSignedUrl({
          fileId: uploadedImage.id,
        });
        
        console.log("Got signed URL for image:", signedUrl.url.substring(0, 50) + "...");
        
        // 3. Process the image with OCR
        const ocrResponse = await mistralClient.ocr.process({
          model: "mistral-ocr-latest",
          document: {
            type: "image_url",
            imageUrl: signedUrl.url,
          }
        }) as MistralOCRResponse;
        
        console.log("OCR processing complete, extracting text content");
        console.log("OCR response structure:", JSON.stringify(ocrResponse, null, 2).substring(0, 500) + "...");
        
        // 4. Extract text from OCR response with improved handling
        if (ocrResponse) {
          // Try to extract from different possible response structures
          if (ocrResponse.pages && Array.isArray(ocrResponse.pages)) {
            console.log(`Found ${ocrResponse.pages.length} pages in OCR response`);
            
            textContent = ocrResponse.pages.map((page, index) => {
              console.log(`Processing page ${index + 1}, type: ${typeof page}`);
              
              if (typeof page === 'string') {
                return page;
              } else if (typeof page === 'object' && page !== null) {
                // Log available properties on this page object
                console.log(`Page ${index + 1} properties:`, Object.keys(page));
                
                // Check for different property names that might contain text
                if (page.text) return page.text;
                if (page.content) return page.content;
                if (page.markdown) return page.markdown;
                
                // Try to stringify the entire object if we couldn't find specific text properties
                try {
                  return JSON.stringify(page);
                } catch (e) {
                  console.error(`Error stringifying page ${index + 1}:`, e);
                  return '';
                }
              }
              return '';
            }).join("\n\n");
            
            console.log(`Extracted ${textContent.length} characters of text content`);
            
            // Fallback if text is still empty
            if (!textContent.trim()) {
              console.log("No text extracted from pages array. Trying alternate approaches.");
              
              // Check if there's a simpler text property directly on the response
              if (ocrResponse.text) {
                textContent = ocrResponse.text;
                console.log("Found text directly on OCR response");
              } else if (typeof ocrResponse === 'string') {
                textContent = ocrResponse;
                console.log("OCR response is a string, using directly");
              } else {
                // Last resort: stringify the response object, but clean up to extract only text
                try {
                  const fullResponseString = JSON.stringify(ocrResponse);
                  console.log("Using full response string as fallback");
                  textContent = fullResponseString;
                } catch (e) {
                  console.error("Error converting OCR response to string:", e);
                }
              }
            }
          } else {
            console.log("OCR response does not have a pages array, structure:", 
                        typeof ocrResponse, 
                        Object.keys(ocrResponse));
            
            // Try to extract text from other potential response formats
            if (ocrResponse.text) {
              textContent = ocrResponse.text;
            } else if (ocrResponse.content) {
              textContent = ocrResponse.content;
            } else if (ocrResponse.markdown) {
              textContent = ocrResponse.markdown;
            } else {
              // Try to stringify response for inspection
              try {
                textContent = JSON.stringify(ocrResponse);
                console.log("Using stringified response:", textContent.substring(0, 100) + "...");
              } catch (e) {
                console.error("Error stringifying OCR response:", e);
                textContent = "Error extracting text from OCR response";
              }
            }
          }
        } else {
          console.error("OCR response is null or undefined");
          textContent = "Error: Could not obtain OCR response";
        }
        
        // 5. Extract token usage information
        if (ocrResponse?.usageInfo) {
          if (ocrResponse.usageInfo.totalTokens) {
            tokensUsed = ocrResponse.usageInfo.totalTokens;
          } else if (ocrResponse.usageInfo.total_tokens) {
            tokensUsed = ocrResponse.usageInfo.total_tokens;
          }
          console.log("Token usage:", tokensUsed);
        } else {
          console.log("No usage info available in OCR response");
        }
        
        // Optional: Delete the uploaded file to clean up
        // Commented out for now to prevent potential issues
        /*
        try {
          await mistralClient.files.delete({
            fileId: uploadedImage.id,
          });
          console.log("Deleted uploaded file from Mistral");
        } catch (deleteError) {
          console.error("Failed to delete uploaded file:", deleteError);
          // Continue processing even if deletion fails
        }
        */
      }
    } catch (ocrError: unknown) {
      const error = ocrError as ProcessingError;
      console.error("Error during OCR processing:", error);
      
      // Update file status to error
      await db
        .update(uploadedFiles)
        .set({
          status: "error",
          error: error.message || "OCR processing failed",
          updatedAt: new Date(),
        })
        .where(eq(uploadedFiles.id, fileId));
      
      throw error; // Rethrow to be caught by the outer try-catch
    }

    // Final check to ensure we have text content
    if (!textContent || textContent.trim() === '') {
      console.warn("No text content was extracted, using fallback placeholder");
      textContent = "⚠️ OCR processing completed, but no text could be extracted from this document.";
    }

    // Update database with results
    await db
      .update(uploadedFiles)
      .set({
        status: "completed",
        textContent: textContent,
        tokensUsed: tokensUsed,
        updatedAt: new Date(),
      })
      .where(eq(uploadedFiles.id, fileId));

    // Update user's token usage if user management is enabled
    if (process.env.ENABLE_USER_MANAGEMENT === "true") {
      console.log("Incrementing token usage for user:", userId, tokensUsed);
      try {
        await incrementAndLogTokenUsage(userId, tokensUsed);
      } catch (error) {
        console.error("Error updating token usage:", error);
        // Continue processing - don't fail the request if token tracking fails
      }
    }

    return NextResponse.json({
      success: true,
      text: textContent,
    });
  } catch (error: unknown) {
    const err = error as ProcessingError;
    console.error("Processing error:", err);

    // Update file status to error if we have a fileId
    if (fileId !== null) {
      await db
        .update(uploadedFiles)
        .set({
          status: "error",
          error: err.message,
          updatedAt: new Date(),
        })
        .where(eq(uploadedFiles.id, fileId));
    }

    return NextResponse.json(
      { error: "Failed to process file" },
      { status: 500 }
    );
  }
}
