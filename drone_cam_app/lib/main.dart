import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const DroneCamApp());
}

class DroneCamApp extends StatelessWidget {
  const DroneCamApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DroneCam Sensor',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.deepPurple,
        scaffoldBackgroundColor: const Color(0xFF0B0F19),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1E1B4B),
          centerTitle: true,
        ),
      ),
      home: const DroneScreen(),
    );
  }
}

class DroneScreen extends StatefulWidget {
  const DroneScreen({super.key});

  @override
  State<DroneScreen> createState() => _DroneScreenState();
}

class _DroneScreenState extends State<DroneScreen> {
  File? _image;
  final ImagePicker _picker = ImagePicker();
  final TextEditingController _ipController = TextEditingController(text: '192.168.1.X:5001');
  
  bool _isProcessing = false;
  bool _isTransmitting = false;
  
  Uint8List? _mapImageBytes;
  Uint8List? _encryptedImageBytes;

  Future<void> _captureImage() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.camera);
    if (photo != null) {
      setState(() {
        _image = File(photo.path);
        _mapImageBytes = null;
        _encryptedImageBytes = null;
      });
    }
  }

  Future<void> _pickGallery() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _image = File(image.path);
        _mapImageBytes = null;
        _encryptedImageBytes = null;
      });
    }
  }

  Future<void> _processOnDrone() async {
    if (_image == null) return;
    
    setState(() {
      _isProcessing = true;
    });

    try {
      final ip = _ipController.text.trim();
      final uri = Uri.parse('http://$ip/process_on_drone');
      
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('image', _image!.path));
      
      var response = await request.send();
      
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        var jsonResponse = json.decode(responseData);
        
        setState(() {
          _mapImageBytes = base64Decode(jsonResponse['map_image']);
          _encryptedImageBytes = base64Decode(jsonResponse['encrypted_image']);
        });
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Drone Edge Processing Complete!'), backgroundColor: Colors.green),
          );
        }
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('❌ Error: ${response.statusCode}'), backgroundColor: Colors.red),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('❌ Connection Failed: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }
  
  Future<void> _transmitToGroundStation() async {
    if (_image == null) return;
    
    setState(() {
      _isTransmitting = true;
    });

    try {
      final ip = _ipController.text.trim();
      final uri = Uri.parse('http://$ip/upload');
      
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('image', _image!.path));
      
      var response = await request.send();
      
      if (response.statusCode == 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('✅ Encrypted Payload Transmitted!'), backgroundColor: Colors.blue),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('❌ Connection Failed: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isTransmitting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Drone Brain 🚁', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // IP Configuration
            Card(
              color: const Color(0xFF1E1E2C),
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: TextField(
                  controller: _ipController,
                  decoration: const InputDecoration(
                    labelText: 'API Gateway IP',
                    prefixIcon: Icon(Icons.wifi),
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 15),
            
            // Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _captureImage,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Capture'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blueAccent,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _pickGallery,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.teal,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            
            // Main Display Area
            if (_image != null && _mapImageBytes == null) ...[
                const Text("Raw Sensor Data (Plaintext)", style: TextStyle(color: Colors.grey, fontSize: 16)),
                const SizedBox(height: 5),
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.file(_image!, height: 250, fit: BoxFit.cover),
                ),
                const SizedBox(height: 20),
                SizedBox(
                  height: 50,
                  child: ElevatedButton.icon(
                    onPressed: _isProcessing ? null : _processOnDrone,
                    icon: _isProcessing 
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Icon(Icons.memory),
                    label: const Text('Execute ML & Encryption', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.purpleAccent),
                  ),
                ),
            ],
            
            if (_mapImageBytes != null && _encryptedImageBytes != null) ...[
                const Text("Drone ML Classification Map", style: TextStyle(color: Colors.grey, fontSize: 16)),
                const SizedBox(height: 5),
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.memory(_mapImageBytes!, height: 200, fit: BoxFit.cover),
                ),
                const SizedBox(height: 15),
                const Text("Encrypted Payload (Ready to Transmit)", style: TextStyle(color: Colors.redAccent, fontSize: 16, fontWeight: FontWeight.bold)),
                const SizedBox(height: 5),
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.memory(_encryptedImageBytes!, height: 200, fit: BoxFit.cover),
                ),
                const SizedBox(height: 20),
                SizedBox(
                  height: 60,
                  child: ElevatedButton.icon(
                    onPressed: _isTransmitting ? null : _transmitToGroundStation,
                    icon: _isTransmitting 
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Icon(Icons.cell_tower, size: 28),
                    label: Text(
                      _isTransmitting ? 'Transmitting...' : 'TRANSMIT ENCRYPTED PAYLOAD',
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}
