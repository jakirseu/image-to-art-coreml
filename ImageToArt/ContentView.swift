 

import SwiftUI
import CoreML
import Vision
import UIKit

struct ContentView: View {
    @State private var image: UIImage? = nil
    @State private var output: UIImage? = nil
    @State private var isImagePickerPresented: Bool = false
    
    
    
    var body: some View {
        VStack {
            ScrollView{
                
                
                if let image = image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .padding()
                    
                    HStack{
                        Button("Make Art") {
                            makeArt(image)
                        }
                  
                     
                        
                    }
                }
                
                if let output = output{
                    Image(uiImage: output)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .padding()
                    
                    Button("Save"){
                        UIImageWriteToSavedPhotosAlbum(output, nil, nil, nil)
                    }
                }
                
                Button("Pick an Image") {
                    isImagePickerPresented = true
                }
            }
            .padding()
            
            
        }
        .sheet(isPresented: $isImagePickerPresented, content: {
            ImagePicker(selectedImage: $image, sourceType: .photoLibrary)
            
        })
    }
    
    // Function to Make Art from the image using Core ML
    func makeArt(_ image: UIImage) {
        do {
            let modelConfiguration = MLModelConfiguration()
            // Load yoru model
            let model = try VNCoreMLModel(for: RickshawArt(configuration: modelConfiguration).model)
            
            let request = VNCoreMLRequest(model: model) { request, error in
                if let results = request.results as? [VNPixelBufferObservation],
                   let pixelBuffer = results.first?.pixelBuffer {
                    
                    // Convert the pixel buffer to CIImage
                    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                    let context = CIContext()
                    
                    // Create CGImage
                    if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                        // Convert CGImage to UIImage
                        self.output = UIImage(cgImage: cgImage)
                        // Use the processed UIImage (uiImage)
                    }
                } else {
                    print("No results or invalid format")
                }
            }
            
            // Load the image from the Assets
            if  let cgImage = image.cgImage {
                let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                try handler.perform([request])
            }
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    }
    
    // ImagePicker helper to allow the user to pick an image
    struct ImagePicker: UIViewControllerRepresentable {
        @Binding var selectedImage: UIImage?
        @Environment(\.presentationMode) var presentationMode
        var sourceType: UIImagePickerController.SourceType = .photoLibrary
        
        class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
            let parent: ImagePicker
            
            init(parent: ImagePicker) {
                self.parent = parent
            }
            
            func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
                if let image = info[.originalImage] as? UIImage {
                    parent.selectedImage = image
                }
                parent.presentationMode.wrappedValue.dismiss()
            }
            
            func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
                parent.presentationMode.wrappedValue.dismiss()
            }
        }
        
        func makeCoordinator() -> Coordinator {
            Coordinator(parent: self)
        }
        
        func makeUIViewController(context: UIViewControllerRepresentableContext<ImagePicker>) -> UIImagePickerController {
            let picker = UIImagePickerController()
            picker.delegate = context.coordinator
            picker.sourceType = sourceType
            return picker
        }
        
        func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<ImagePicker>) {}
    }
    
}
#Preview {
    ContentView()
}
