async function predictCrop() {

    const data = {

        N: Number(document.getElementById("N").value),

        P: Number(document.getElementById("P").value),

        K: Number(document.getElementById("K").value),

        temperature:
            Number(document.getElementById("temperature").value),

        humidity:
            Number(document.getElementById("humidity").value),

        ph:
            Number(document.getElementById("ph").value),

        rainfall:
            Number(document.getElementById("rainfall").value)
    };


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const result = await response.json();


        if (result.success) {

            document.getElementById("result").innerHTML =
                "🌱 Recommended Crop: <b>" +
                result.crop +
                "</b><br><br>" +
                result.message;

        }

        else {

            document.getElementById("result").innerHTML =
                "❌ Error: " +
                result.error;

        }


    }

    catch (error) {

        document.getElementById("result").innerHTML =
            "❌ Could not connect to Flask backend.";

        console.error(error);
    }
}


async function sendMessage() {

    const input =
        document.getElementById("user-message");

    const message =
        input.value.trim();


    if (!message) {

        return;
    }


    const chatBox =
        document.getElementById("chat-box");


    chatBox.innerHTML +=
        `<p><b>You:</b> ${message}</p>`;


    input.value = "";


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/chat",
            {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    message: message
                })
            }
        );


        const result =
            await response.json();


        if (result.answer) {

            chatBox.innerHTML +=
                `<p><b>🤖 AI:</b> ${result.answer}</p>`;

        }

        else {

            chatBox.innerHTML +=
                `<p><b>❌ Error:</b> ${result.error}</p>`;
        }


        chatBox.scrollTop =
            chatBox.scrollHeight;


    }

    catch (error) {

        chatBox.innerHTML +=
            `<p><b>❌ Error:</b> AI server unavailable.</p>`;
    }
}