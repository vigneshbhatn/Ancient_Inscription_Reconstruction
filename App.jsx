import React, { useState } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [processed, setProcessed] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await axios.post("http://127.0.0.1:8000/upload/", formData, {
      responseType: "blob"
    });

    const imgUrl = URL.createObjectURL(res.data);
    setProcessed(imgUrl);
  };

  return (
    <div className="p-6 flex flex-col items-center">
      <h1 className="text-xl font-bold mb-4">Kannada Inscription Preprocessor</h1>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4"
      />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        Upload & Process
      </button>

      {processed && (
        <div className="mt-6">
          <h2 className="mb-2 font-semibold">Processed Image:</h2>
          <img src={processed} alt="Processed" className="border rounded shadow-lg"/>
        </div>
      )}
    </div>
  );
}
