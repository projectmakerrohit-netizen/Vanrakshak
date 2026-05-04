require("dotenv").config();

const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");
const TelegramBot = require("node-telegram-bot-api");

const { createClient } = require("@supabase/supabase-js");

const app = express();
app.use(express.json());

/* =========================
   TELEGRAM
========================= */
const bot = new TelegramBot(
  process.env.TELEGRAM_BOT_TOKEN,
  { polling: false }
);

async function sendTelegramAlert(message) {
  try {
    await bot.sendMessage(
      process.env.TELEGRAM_CHAT_ID,
      message
    );

    console.log("Telegram alert sent");
  } catch (err) {
    console.error("Telegram Error:", err.message);
  }
}

/* =========================
   SUPABASE
========================= */
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

/* =========================
   UPLOAD DIR
========================= */
const uploadDir = path.join(__dirname, "uploads");

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

/* =========================
   MULTER
========================= */
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),

  filename: (req, file, cb) => {
    cb(
      null,
      Date.now() + path.extname(file.originalname)
    );
  }
});

const upload = multer({ storage });

/* =========================
   ROOT
========================= */
app.get("/", (req, res) => {
  res.json({
    service: "Vanraksha Backend",
    status: "running"
  });
});

/* =========================
   ALERTS
========================= */
app.get("/alerts", async (req, res) => {
  try {
    const { data, error } = await supabase
      .from("alerts")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) throw error;

    res.json({
      success: true,
      alerts: data
    });

  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message
    });
  }
});

/* =========================
   UPLOAD PIPELINE
========================= */
app.post("/upload", upload.single("file"), async (req, res) => {
  let localPath = null;

  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "No file uploaded"
      });
    }

    localPath = req.file.path;
    const fileName = req.file.filename;

    /* 1) upload storage */
    const fileBuffer = fs.readFileSync(localPath);

    const { error: uploadError } =
      await supabase.storage
        .from("evidence")
        .upload(fileName, fileBuffer, {
          contentType: req.file.mimetype,
          upsert: true
        });

    if (uploadError) throw uploadError;

    const { data: urlData } =
      supabase.storage
        .from("evidence")
        .getPublicUrl(fileName);

    const imageUrl = urlData.publicUrl;

    /* 2) ML analyze */
    const form = new FormData();

    form.append(
      "file",
      fs.createReadStream(localPath)
    );

    const mlResponse = await axios.post(
      "http://127.0.0.1:8000/analyze-image",
      form,
      {
        headers: form.getHeaders()
      }
    );

    const ml = mlResponse.data;

    /* 3) save alert */
    const alertPayload = {
      type: ml.type,
      confidence: ml.confidence,
      risk_level: ml.risk,
      objects: ml.objects,
      image_url: imageUrl,
      latitude: 22.57,
      longitude: 88.36
    };

    const { data: savedAlert, error: dbError } =
      await supabase
        .from("alerts")
        .insert(alertPayload)
        .select()
        .single();

    if (dbError) throw dbError;

    /* 4) PHONE ALERT */
    if (ml.risk === "high") {
      const msg =
`🚨 VANRAKSHA ALERT 🚨

Type: ${ml.type}
Objects: ${ml.objects.join(", ")}
Gun: ${ml.gun_detected}
Confidence: ${ml.confidence}
Risk: ${ml.risk}

Latitude: 22.57
Longitude: 88.36

Evidence:
${imageUrl}`;

      await sendTelegramAlert(msg);
    }

    /* cleanup */
    fs.unlinkSync(localPath);

    res.json({
      success: true,
      message: "Analyzed + Alerted",
      ml_result: ml,
      alert: savedAlert
    });

  } catch (err) {
    console.error(err);

    if (localPath && fs.existsSync(localPath)) {
      fs.unlinkSync(localPath);
    }

    res.status(500).json({
      success: false,
      error: err.message
    });
  }
});

/* =========================
   START
========================= */
const PORT = 5000;

app.listen(PORT, () => {
  console.log(
    `Backend running on http://localhost:${PORT}`
  );
});
