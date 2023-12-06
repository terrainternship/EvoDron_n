"use client";

import React, { useState, useEffect } from "react";
import "tailwindcss/tailwind.css";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [modelUsed, setModelUsed] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<string>("No model selected");
  const [modelDescription, setModelDescription] = useState<string | null>(null);
  const [modelPhoto, setModelPhoto] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState(""); // Initialize with the original image source
  const [originalImageSrc, setOriginalImageSrc] = useState(""); // Initialize with the original image source
  const [resultedImageSrc, setResultedImageSrc] = useState(""); // Initialize with an empty string
  const [isOriginal, setIsOriginal] = useState(true);

  const [isVideoThumbnail, setIsVideoThumbnail] = useState(false);

  const [gifSrc, setGifSrc] = useState<string | null>(null);

  const [serverStatus, setServerStatus] = useState<string | null>(null);
  const [isLoadingServerStatus, setIsLoadingServerStatus] = useState<
    boolean | null
  >(null);
  const [lastImageFile, setLastImageFile] = useState<File | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(true);

  const [isLoadingModelData, setIsLoadingModelData] = useState(false);
  const [namesString, setNamesString] = useState("");
  const [isResultReceived, setIsResultReceived] = useState(false);
  interface Frame {
    frame_number: number;
    time_in_seconds: number;
    annotated_image: string;
    frame_results: any[];
    detection_results: {
      "Total # of instances": number;
      Types: string[];
      "Total # of classes": number;
      "Area by type": Record<string, { area: number; area_percentage: number }>;
      instances: Array<{
        name: string;
        area: number;
        area_percentage: number;
      }>;
    };
  }

  interface Instance {
    name: string;
    area: number;
    area_percentage: string;
  }

  interface DetectionResults {
    "Total # of instances": number;
    "Total # of classes": number;
    "Area by type": Record<string, { area: number; area_percentage: string }>;
    instances: Instance[];
  }

  interface DetectionSummary {
    "Total # of instances": number;
    "Total # of classes": number;
    "Total area by type": Record<string, { area: number }>;
    "Average % area by type": Record<string, { percentage_area: string }>;
  }
  interface DataType {
    type: string;
    image: string;
    model_used: string;
    frames?: Frame[];
    detection_results?: DetectionResults;
    detection_summary?: DetectionSummary;
  }
  interface Image {
    filename: string;
    is_video: boolean;
    thumbnail?: string;
  }

  const [sharedImages, setSharedImages] = useState<Image[]>([]);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [data, setData] = useState<DataType | null>(null);

  const [currentImage, setCurrentImage] = useState<string | null>(null);

  const [currentMediaIndex, setCurrentMediaIndex] = useState<number | null>(
    null
  );

  const imageInputRef = React.useRef<HTMLInputElement>(null);
  const [projectStructure, setProjectStructure] = useState<any>(null);

  const [isUploadPopupOpen, setIsUploadPopupOpen] = useState(false);
  const [description, setDescription] = useState("");

  const [models, setModels] = useState<string[]>([]);
  const [diskContent, setDiskContent] = useState<string[]>([]);

  const SERVER_URL = process.env.SERVER_URL;

  const GETMODELS_URL = `${SERVER_URL}/models`;
  const PREDICT_URL = `${SERVER_URL}/predict`;
  const UPLOAD_MODEL_URL = `${SERVER_URL}/upload_model`;
  const CURRENT_MODEL_URL = `${SERVER_URL}/current_model`;
  const DOWNLOAD_MODEL_URL = `${SERVER_URL}/download_model`;
  const SHARED_IMAGES_URL = `${SERVER_URL}/shared_images`;
  const SELECT_MODEL_URL = `${SERVER_URL}/select_model`;
  const MODEL_INFO_URL = `${SERVER_URL}/model_info/`;

  const fetchModels = async () => {
    try {
      const response = await fetch(GETMODELS_URL, {
        credentials: "include", // Include cookies
      });

      const { models: modelDirs } = await response.json();
      setModels(modelDirs);
    } catch (error) {
      console.error("Failed to fetch models:", error);
    }
  };

  const fetchSharedImages = async () => {
    try {
      const response = await fetch(SHARED_IMAGES_URL, {
        credentials: "include", // Include cookies
      });
      const data = await response.json();
      setSharedImages(data.images);
    } catch (error) {
      console.error("Failed to fetch shared images:", error);
    }
  };

  function getVideoThumbnail(file: Blob) {
    return new Promise((resolve, reject) => {
      // Create a video element
      const video = document.createElement("video");

      // When the metadata has been loaded, set the time to the thumbnail frame time
      video.onloadedmetadata = function () {
        video.currentTime = 0;
      };

      // When the video has seeked to the correct time, draw the frame on a canvas
      video.onseeked = function () {
        // Create a canvas and draw the video frame on it
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          reject("Could not create 2D context");
          return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas to a data URL
        const thumbnail = canvas.toDataURL("image/jpeg");

        // Resolve the promise with the thumbnail
        resolve(thumbnail);
      };

      // Set the video source
      video.src = URL.createObjectURL(file);

      // Catch any errors
      video.onerror = function (err) {
        reject(err);
      };
    });
  }

  const resetPage = () => {
    setIsResultReceived(false);
    setCurrentImage(null);
    setGifSrc(null);
  }

  const handlePredict = async () => {
    if (!currentImage && currentMediaIndex === null) {
      setError("No image or media selected for prediction.");
      return;
    }

    setIsLoading(true);
    setData(null);
    setError(null);

    setIsResultReceived(false);

    try {
      let formData = new FormData();

      if (currentMediaIndex !== null) {
        // If currentMediaIndex is not null, it's an index in sharedImages
        formData.append("mediaIndex", currentMediaIndex.toString());
      } else if (uploadedFile) {
        // If uploadedFile is not null, it's an uploaded image or video
        let blob = uploadedFile;
        let extension = "";
        let type = "";
        if (blob.type === "image/jpeg") {
          extension = ".jpg";
          type = "image";
        } else if (blob.type === "image/png") {
          extension = ".png";
          type = "image";
        } else if (blob.type === "video/mp4") {
          extension = ".mp4";
          type = "video";
        }

        const filename = `file${extension}`;
        formData.append("file", blob, filename);
        formData.append("type", type);
      }





      const predictResponse = await fetch(PREDICT_URL, {
        method: "POST",
        body: formData,
        credentials: "include", // Include cookies
       
      });

 
      console.log("3");
      if (!predictResponse.ok) {
        throw new Error(`HTTP error! status: ${predictResponse.status}`);
      }

      const responseData = await predictResponse.json();
      console.log(responseData);
      if (responseData.type === "image") {
        setModelUsed(responseData.model_used); // Store the model used in state
        setData(responseData);
      } else if (responseData.type === "video") {

        if (responseData.gif) {
          setGifSrc(`data:image/gif;base64,${responseData.gif}`);
        }
        // Handle video results
        // responseData.results is now an array of objects, each with a 'frame', 'results', and 'annotated_image' property
        setData(responseData);
        setModelUsed(responseData.model_used); // Store the model used in state
      }

      // Assuming responseData.image is the base64 encoded image
      const base64Image = `data:image/jpeg;base64,${responseData.image}`;

      // Store the original image source before making the prediction
      if (currentImage) {
        setOriginalImageSrc(currentImage);
      }

      setResultedImageSrc(base64Image);
      setImageSrc(base64Image);
      setIsOriginal(false);

      setData(responseData); // Set data to the response data
      setIsResultReceived(true);

      // Assuming the response data has a property 'image' which holds the image data
    } catch (error) {
      console.error();
      let errorMessage = "An error occurred";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);

      setError(null);
    }
  };

  const toggleImage = () => {
    if (isOriginal) {
      setImageSrc(resultedImageSrc); // Replace with the resulted image source
    } else {
      setImageSrc(originalImageSrc); // Replace with the original image source
    }
    setIsOriginal(!isOriginal);
  };

/*   const fetchProjectStructure = async () => {
    try {
      const response = await fetch(`${SERVER_URL}/project_structure`);
      const data = await response.json();
      setProjectStructure(data);
    } catch (error) {
      console.error('Failed to fetch project structure:', error);
    }
  }; */

  const handleMediaUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target?.files?.[0]) {
      const file = event.target.files[0];
      const fileExtension = file.name.split(".").pop()?.toLowerCase();

      // Check if the file size is over 10MB
      if (file.size > 10 * 1024 * 1024) {
        setError("File size should not exceed 10MB.");
        return;
      }

      setLastImageFile(file);
      setCurrentMediaIndex(null);
      setError(null);

      // If the file is a video, create a thumbnail and set it as the current image
      if (fileExtension === "mp4") {
        try {
          const thumbnail = await getVideoThumbnail(file);
          setCurrentImage(thumbnail as string);
          setIsVideoThumbnail(true); // Set isVideoThumbnail to true
        } catch (err) {
          console.error("Failed to create video thumbnail:", err);
        }
      } else {
        setIsVideoThumbnail(false);
        const readerForDisplay = new FileReader();
        readerForDisplay.onload = (e) => {
          // Set the uploaded image as the current image
          setCurrentImage(e.target?.result as string);
        };
        readerForDisplay.readAsDataURL(file);
      }

      setUploadedFile(file);
    }
  };

  const handleModelChange = async (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    setIsLoadingModelData(true);
    let selectedModel = event.target.value;

    console.log(selectedModel);

    try {
      const response = await fetch(
        `${SELECT_MODEL_URL}?model_name=${encodeURIComponent(selectedModel)}`,
        {
          method: "POST",
          credentials: "include",
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log(data.message);
      setCurrentModel(selectedModel);
      await fetchModels();
      console.log(selectedModel);
      fetchModelInfo(selectedModel); // Update currentModel with selectedModel
    } catch (error) {
      console.error("Failed to select model:", error);
    }
    setIsLoadingModelData(false);
  };

  const fetchModelInfo = async (modelName: string) => {
    try {
      const response = await fetch(
        `${MODEL_INFO_URL}${encodeURIComponent(modelName)}`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setModelDescription(data.description);
      if (data.photo_url) {
        setModelPhoto(`${SERVER_URL}${data.photo_url}`);
      } else {
        setModelPhoto(null);
      }
    } catch (error) {
      console.error("Failed to fetch model info:", error);
    }
  };

  const handleModelUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const modelFileInput = event.currentTarget.elements.namedItem(
      "model_file"
    ) as HTMLInputElement;
    const modelFile = modelFileInput.files?.[0];
    const modelFileExtension = modelFile?.name.split(".").pop()?.toLowerCase();

    if (modelFile && modelFileExtension !== "pt") {
      setError("Only .pt files are accepted for the model.");
      modelFileInput.value = ""; // Clear the file input
      return;
    }

    const photoFileInput = event.currentTarget.elements.namedItem(
      "photo"
    ) as HTMLInputElement;
    const photoFile = photoFileInput.files?.[0];
    const photoFileExtension = photoFile?.name.split(".").pop()?.toLowerCase();

    if (
      photoFile &&
      photoFileExtension !== "jpg" &&
      photoFileExtension !== "png"
    ) {
      setError("Only JPG and PNG images are accepted for the photo.");
      photoFileInput.value = ""; // Clear the file input
      return;
    }

    const formData = new FormData(event.currentTarget);
    formData.append("description", description);

    setIsLoading(true);
    setError(null);

    try {
      console.log("Sending model to server...");
      const response = await fetch(UPLOAD_MODEL_URL, {
        method: "POST",
        body: formData,
        credentials: "include", // Include cookies
      });

      // Get the model name from the response
      const data = await response.json();
      console.log("Response data:", data);
      if (data.model_name) {
        console.log("Model name:", data.model_name);
        setCurrentModel(data.model_name);
        console.log("Current model:", currentModel);

        // Select the uploaded model
        await handleModelChange({
          target: { value: data.model_name },
        } as React.ChangeEvent<HTMLSelectElement>);
      }

      // Fetch the models
      await fetchModels();
    } catch (error) {
      let errorMessage = "An error occurred";
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setIsUploadPopupOpen(false); // Close the popup
    }
  };

  // This function fetches the current model from the server
  const getCurrentModel = async () => {
    setIsLoadingModel(true);
    const response = await fetch(CURRENT_MODEL_URL, {
      credentials: "include", // Include cookies
    });
    const data = await response.json();
    if (data.model_used) {
      setCurrentModel(data.model_used);
      // Fetch the model info for the current model
      fetchModelInfo(data.model_used);
    } else {
      setCurrentModel("No model selected");
    }
    setIsLoadingModel(false);
  };

  useEffect(() => {
    getCurrentModel();
  }, []);

  const downloadModel = () => {
    window.location.href = DOWNLOAD_MODEL_URL;
  };

  useEffect(() => {}, [currentModel]);

  useEffect(() => {
    fetchModels();
  }, []);

  useEffect(() => {
    fetchSharedImages();
  }, []);

  // Check server status when the page loads
  useEffect(() => {
    setIsLoadingServerStatus(true);

    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Request timed out")), 5000)
    );

    const request = fetch(CURRENT_MODEL_URL)
      .then((response) => {
        if (response.ok) {
          setServerStatus("online");
        } else {
          setServerStatus("offline");
        }
      })
      .catch((error) => {
        setError("Failed to check server status: " + error.message);
      });

    Promise.race([request, timeout])
      .catch((error) => {
        setError("Failed to check server status: " + error.message);
        setServerStatus("offline");
      })
      .finally(() => {
        setIsLoadingServerStatus(false);
      });
  }, [CURRENT_MODEL_URL]);

  const getServerStatusColor = () => {
    if (isLoadingServerStatus) return "yellow";
    if (serverStatus === "online") return "lightgreen";
    return "red";
  };

  const getServerStatusText = () => {
    if (isLoadingServerStatus) return "Loading server status...";
    if (serverStatus === "online") return "Online";
    return "Offline, or out of memory. Try reloading in a minute.";
  };

  return (
    <div className="container mx-auto px-4 mb-32">
      <div className="flex flex-col my-4">
        <h1 className="text-4xl font-bold my-2">Landscape types YOLO tester</h1>
        <p>Run your landscape image or video through a YOLO model</p>
      </div>
      {!isResultReceived && !isLoading && (
        <div className="container mx-auto ">
          {isLoadingServerStatus === null ? null : (
            <p style={{ color: getServerStatusColor() }}>
              Server: {getServerStatusText()}
            </p>
          )}

          {isUploadPopupOpen && (
            <div className="fixed z-10 inset-0 overflow-y-auto flex items-center justify-center ">
              <div className="bg-gray-500 bg-opacity-75 fixed inset-0"></div>

              <div
                className="bg-gray-900 rounded-lg text-left text-white overflow-hidden shadow-xl transform transition-all sm:w-full sm:max-w-lg border-2 "
                role="dialog"
                aria-modal="true"
                aria-labelledby="modal-headline"
              >
                <div className=" p-4 sm:p-6">
                  <h3 className="text-lg leading-6 font-medium ">
                    Upload a new model
                  </h3>
                  {error && <p className="text-red-500">{error}</p>}
                  <form onSubmit={handleModelUpload} className="mt-4">
                    <div>
                      <label
                        htmlFor="model-file"
                        className="block text-sm font-medium text-gray-400"
                      >
                        Model file (.pt):
                      </label>
                      <input
                        type="file"
                        id="model-file"
                        name="model_file"
                        required
                        className="mt-1 block w-full"
                      />
                    </div>
                    <div className="mt-4">
                      <label
                        htmlFor="photo"
                        className="block text-sm font-medium text-gray-400"
                      >
                        Photo (optional):
                      </label>
                      <input
                        type="file"
                        id="photo"
                        name="photo"
                        className="mt-1 block w-full"
                      />
                    </div>
                    <div className="mt-4">
                      <label
                        htmlFor="description"
                        className="block text-sm font-medium text-gray-400"
                      >
                        Description (optional):
                      </label>
                      <textarea
                        id="description"
                        name="description"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        className="mt-1 block w-full border text-black border-gray-300 rounded-md"
                      />
                    </div>
                    <div className="mt-4 flex justify-end">
                      <button
                        type="button"
                        onClick={() => setIsUploadPopupOpen(false)}
                        className="mr-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="text-sm font-medium text-white bg-teal-700 hover:bg-teal-900  px-4 py-2 rounded "
                      >
                        Upload
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          )}

          <hr className="my-4 border-gray-700" />

          <div className="flex flex-col">
            <div className="flex flex-col md:flex-row">
              <div className="w-full md:w-1/2 p-2">
                <p className="text-xl mt-2">Select a Model (2.7 без масок, в данное время не поддерживается)</p>
                <select
  id="model-select"
  onChange={async (event) => await handleModelChange(event)}
  disabled={isLoading}
  value={currentModel}
  style={{ width: "80%", color: "black" }}
>
  {models.map((modelDir: string, index: number) => (
    <option key={index} value={modelDir} disabled={index === 0}>
      {index === 0 ? `${modelDir} (модель пока не поддерживается)` : modelDir}
    </option>
  ))}
</select>
                <div className="mt-2">
                  <h2>Description</h2>
                  <p>
                    {isLoadingModelData
                      ? "Loading model description..."
                      : modelDescription || "No model description"}
                  </p>
                </div>
              </div>
              <div className="w-full md:w-1/2 p-2 flex flex-col items-center">
                {isLoadingModelData ? (
                  <p>Loading model image...</p>
                ) : modelPhoto && modelPhoto.trim() !== "" ? (
                  <div>
                    <div style={{ display: "flex", justifyContent: "center" }}>
                      <img
                        src={modelPhoto}
                        alt="Model image"
                        style={{ maxHeight: "400px" }}
                      />
                    </div>
                    <button
                      onClick={downloadModel}
                      className="bg-teal-700 hover:bg-teal-900 text-white py-2 px-4 rounded mt-4"
                    >
                      Download model
                    </button>
                  </div>
                ) : (
                  <p>No model photo</p>
                )}
              </div>
            </div>
            {/* <div className="flex flex-col items-end max-w-1/2">
            <p>Current model: </p>{" "}
            <p className="text-white font-bold">
              {isLoadingModel ? "Loading model..." : currentModel}
            </p>
            <a
              href="https://drive.google.com/drive/folders/1IY27vNFNr5GC9clNgasYnij9ssSd84yV"
              className="text-blue-500 hover:underline mt-2"
            >
              Source
            </a> */}{" "}
            {/* Replace "/your-link-path" and "Your Link Text" with your actual link and text */}
            {/*  <button
              onClick={downloadModel}
              className="bg-teal-700 hover:bg-teal-900 text-white py-2 px-4 rounded mt-4"
            >
              Download model
            </button> */}
            {/* </div> */}
            {/* </div> */}
          </div>

          {/* {error && <p style={{ color: "red" }}>Error: {error}</p>} */}

          <button
            onClick={() => setIsUploadPopupOpen(true)}
            className="bg-teal-700 hover:bg-teal-900 text-white py-2 px-4 rounded mt-4"
          >
            Upload a new model
          </button>

          {/*     <input
        type="file"
        id="model-input"
        onChange={handleModelChangeUpload}
        disabled={isLoading}
        ref={fileInputRef}
      /> */}

          <hr className="my-4 border-gray-700" />
          <p className="text-xl mt-2">Select an image/video for prediction</p>
          <div className="flex flex-col md:flex-row items-start">
            <div className="w-full md:w-1/2">
              <div className="flex flex-col mt-4">
                <div className="flex flex-row flex-wrap">
                  {sharedImages.map((image, index) => (
                    <div
                      key={index}
                      className={`relative w-32 h-32 m-2 transition duration-500 ease-in-out transform hover:scale-105 hover:opacity-50 
                    ${
                      currentMediaIndex === index
                        ? "border-4 border-teal-500"
                        : ""
                    }`}
                      onClick={() => {
                        const mediaIndex = index;
                        setCurrentMediaIndex(mediaIndex);
                        setIsVideoThumbnail(image.is_video); // Set isVideoThumbnail to true if the image is a video
                        setCurrentImage(
                          `${SERVER_URL}/${
                            image.is_video
                              ? "shared_thumbnails"
                              : "shared_images"
                          }/${
                            image.is_video ? image.thumbnail : image.filename
                          }`
                        );
                      }}
                    >
                      <img
                        src={`${SERVER_URL}/${
                          image.is_video ? "shared_thumbnails" : "shared_images"
                        }/${image.is_video ? image.thumbnail : image.filename}`}
                        alt={image.filename}
                        className={`object-cover w-full h-full ${
                          image.is_video ? "filter brightness-50" : ""
                        }`}
                      />
                      {image.is_video && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="currentColor"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            className="w-12 h-12 text-white"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 4v16l15-8-15-8z"
                            />
                          </svg>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex flex-col mt-4">
                <p className="text-xl mt-2">Or upload a new image/video</p>
                <div className="flex justify-between items-start mt-2">
                  <div>
                    <input
                      type="file"
                      id="image-input"
                      onChange={handleMediaUpload}
                      disabled={isLoading}
                      ref={imageInputRef}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="w-full md:w-1/2 mt-5 flex flex-col items-start md:flex-grow ">
              {currentImage && (
                <>
                  <img
                    src={currentImage}
                    alt="Current"
                    className={`object-contain w-full h-full ${
                      isVideoThumbnail ? "filter brightness-50" : ""
                    }`}
                  />
                  {isVideoThumbnail && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        className="w-12 h-12 text-white"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 4v16l15-8-15-8z"
                        />
                      </svg>
                    </div>
                  )}
                </>
              )}
              {currentImage ? (
                <button
                  onClick={handlePredict}
                  className="bg-teal-700 hover:bg-teal-900 text-white py-2 px-4 rounded mt-4"
                >
                  Predict
                </button>
              ) : (
                <p>Select or upload an image for prediction</p>
              )}
            </div>
          </div>
            {/* <div className="flex flex-col mt-4">
        <p className="text-xl mt-2">Check structure</p>

      <button onClick={fetchProjectStructure}>
  Fetch Project Structure
</button>
<pre>{JSON.stringify(projectStructure, null, 2)}</pre>

<button onClick={fetchDiskContent}>
  Fetch Disk Content
</button>
// Display the disk content in a text element
<pre>{JSON.stringify(diskContent, null, 2)}</pre>
</div> */}
        </div>
      )}
      {(isLoading || isResultReceived) && (
        <div className="container mx-auto my-2">
          <button
            type="button"
            className="text-sm font-medium text-white bg-teal-700 hover:bg-teal-900 px-4 py-2 rounded"
            onClick={() => {
              resetPage();
            }}
          >
            &#8592; Back
          </button>

          {isLoading && (
            <div className="flex justify-center items-center">
              <div className="loader mt-12"></div>
            </div>
          )}
          {error && <p style={{ color: "red" }}>Error: {error}</p>}

          {isResultReceived && (
            <div>
              <hr className="my-4 border-gray-700" />
              <p className="text-xl mt-2">Result</p>

              <p className="mb-2">Model used: {modelUsed}</p>

              <div className="flex flex-col">
                <div>
                  <div>
                    <table style={{ tableLayout: "fixed" }}>
                      <thead>
                        <tr>
                          <th
                            style={{ textAlign: "start", paddingRight: "20px" }}
                          >
                            Total instances
                          </th>
                          <th style={{ textAlign: "start" }}>
                            {data?.type === "image"
                              ? data?.detection_results?.[
                                  "Total # of instances"
                                ]
                              : data?.detection_summary?.[
                                  "Total # of instances"
                                ]}
                          </th>
                        </tr>
                        <tr>
                          <th
                            style={{ textAlign: "start", paddingRight: "20px" }}
                          >
                            Total classes
                          </th>
                          <th style={{ textAlign: "start" }}>
                            {data?.type === "image"
                              ? data?.detection_results?.["Total # of classes"]
                              : data?.detection_summary?.["Total # of classes"]}
                          </th>
                        </tr>
                      </thead>
                    </table>
                    {data?.type === "video" && (
                      <div className="mt-4">
                        <h2>Summary for {data?.frames?.length} frames</h2>
                        <table style={{ tableLayout: "fixed", width: "100%" }}>
                          <thead>
                            <tr>
                              <th style={{ textAlign: "start" }}>Type</th>
                              <th style={{ textAlign: "start" }}>
                                Average Area
                              </th>
                              <th style={{ textAlign: "start" }}>
                                Average % Area
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(
                              data?.detection_summary?.[
                                "Average % area by type"
                              ] ?? {}
                            ).map(([key, value], index) => (
                              <tr key={index}>
                                <td style={{ textAlign: "start" }}>{key}</td>
                                <td style={{ textAlign: "start" }}>
                                  {
                                    data?.detection_summary?.[
                                      "Total area by type"
                                    ][key].area
                                  }
                                </td>
                                <td style={{ textAlign: "start" }}>
                                  {value.percentage_area}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
                <div className="md:flex mt-4">
                  <div className="w-full md:w-1/2">
                    {data?.type === "image" && (
                      <div>
                        <h2 className="mb-2">Area by instance</h2>
                        <table style={{ tableLayout: "fixed", width: "100%" }}>
                          <thead>
                            <tr>
                              <th style={{ textAlign: "start" }}>Instance</th>
                              <th style={{ textAlign: "start" }}>Total</th>
                              <th style={{ textAlign: "start" }}>%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {data.detection_results?.instances?.map(
                              (result, index) => (
                                <tr key={index}>
                                  <td style={{ textAlign: "start" }}>
                                    {result.name}
                                  </td>
                                  <td style={{ textAlign: "start" }}>
                                    {result.area}
                                  </td>
                                  <td style={{ textAlign: "start" }}>
                                    {result.area_percentage}
                                  </td>
                                </tr>
                              )
                            )}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>

                  {data?.type === "image" && (
                    <div className="w-full md:w-1/2">
                      <h2 className="mb-2">Area by class</h2>
                      <table style={{ tableLayout: "fixed", width: "100%" }}>
                        <thead>
                          <tr>
                            <th style={{ textAlign: "start" }}>Class</th>
                            <th style={{ textAlign: "start" }}>Total</th>
                            <th style={{ textAlign: "start" }}>%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(
                            data?.detection_results?.["Area by type"] || {}
                          ).map(([key, value], index) => (
                            <tr key={index}>
                              <td style={{ textAlign: "start" }}>{key}</td>
                              <td style={{ textAlign: "start" }}>
                                {value.area}
                              </td>
                              <td style={{ textAlign: "start" }}>
                                {value.area_percentage}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-2">
                <div className="mt-4">
                  {data?.type === "image" && (
                    <div>
                      <hr className="my-4 border-gray-700" />
                      <button
                        onClick={toggleImage}
                        className="text-sm font-medium text-white bg-teal-700 hover:bg-teal-900  px-4 py-2 rounded "
                      >
                        {isOriginal ? "Show Result" : "Show Original"}
                      </button>
                      <img
                        src={imageSrc}
                        alt="Processed"
                        className="w-full h-full mt-4 object-contain"
                      />
                    </div>
                  )}

                  {data?.type === "video" && (
                    <div className="flex flex-wrap">
                      <h2 className="mt-2">Animated</h2>
                      <div className=" w-full flex justify-center">
                      
                      {gifSrc && <img src={gifSrc} alt="Result GIF" />}
                      </div>
                      <div className=" w-full flex">
                      <h2 className="mt-2">Frames</h2>
                      </div>
                      {data?.frames?.map((frame, index) => (
                        <div key={index} className="w-full md:w-1/2 p-2">
                          <img
                            src={`data:image/jpeg;base64,${frame.annotated_image}`}
                            className="w-full h-auto"
                          />
                          <table className="my-4"
                            style={{ tableLayout: "fixed", width: "100%" }}
                          >
                            <thead>
                              <tr>
                                <th style={{ textAlign: "start" }}>Instance</th>
                                <th style={{ textAlign: "start" }}>
                                  Flat area
                                </th>
                                <th style={{ textAlign: "start" }}>%</th>
                              </tr>
                            </thead>
                            <tbody>
                              {frame.detection_results?.instances?.map(
                                (result, index) => (
                                  <tr key={index}>
                                    <td style={{ textAlign: "start" }}>
                                      {result.name}
                                    </td>
                                    <td style={{ textAlign: "start" }}>
                                      {result.area}
                                    </td>
                                    <td style={{ textAlign: "start" }}>
                                      {result.area_percentage}
                                    </td>
                                  </tr>
                                )
                              )}
                            </tbody>
                          </table>
                          <hr className=" border-gray-700" />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
