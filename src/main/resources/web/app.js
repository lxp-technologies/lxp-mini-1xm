"use strict";

const conversation = [];
const transcript = document.querySelector("#transcript");
const form = document.querySelector("#prompt-form");
const sendButton = document.querySelector("#send-button");
const state = document.querySelector("#generation-state");

loadRuntime();
form.addEventListener("submit", sendMessage);
document.querySelector("#clear-button").addEventListener("click", clearConversation);

async function loadRuntime() {
    try {
        const response = await fetch("/health");
        if (!response.ok) throw new Error("HTTP " + response.status);
        const health = await response.json();
        setText("#model-name", health.model);
        setText("#runtime-device", health.device);
        setText("#model-type", health.model_type);
        setText("#context-length", health.context_length + " tokens");
    } catch (error) {
        showState("Impossible de lire les métadonnées: " + error.message, true);
    }
}

async function sendMessage(event) {
    event.preventDefault();
    const input = document.querySelector("#user-message");
    const userMessage = input.value.trim();
    if (!userMessage) return;

    sendButton.disabled = true;
    showState("Génération en cours sur le runtime local...", false);
    try {
        const response = await fetch("/playground/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                system_prompt: document.querySelector("#system-prompt").value,
                messages: conversation,
                user_message: userMessage,
                temperature: Number(document.querySelector("#temperature").value),
                max_new_tokens: Number(document.querySelector("#max-new-tokens").value),
                top_p: Number(document.querySelector("#top-p").value),
                top_k: Number(document.querySelector("#top-k").value)
            })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error ? payload.error.message : "HTTP " + response.status);
        conversation.push({ role: "user", content: userMessage });
        conversation.push({ role: "assistant", content: payload.text });
        input.value = "";
        renderConversation();
        showState(
            "Terminé: " + payload.prompt_tokens + " tokens de prompt, " +
            payload.completion_tokens + " générés.",
            false
        );
    } catch (error) {
        showState(error.message, true);
    } finally {
        sendButton.disabled = false;
        input.focus();
    }
}

function clearConversation() {
    conversation.splice(0, conversation.length);
    renderConversation();
    showState("Conversation effacée dans cet onglet seulement.", false);
}

function renderConversation() {
    transcript.replaceChildren();
    if (conversation.length === 0) {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "Écris un message pour observer une continuation.";
        transcript.append(empty);
        return;
    }
    conversation.forEach(function (message) {
        const article = document.createElement("article");
        article.className = "message " + message.role;
        const role = document.createElement("small");
        role.textContent = message.role === "user" ? "Toi" : "Modèle base";
        const content = document.createElement("span");
        content.textContent = message.content;
        article.append(role, content);
        transcript.append(article);
    });
    transcript.scrollTop = transcript.scrollHeight;
}

function setText(selector, value) {
    document.querySelector(selector).textContent = value;
}

function showState(message, isError) {
    state.textContent = message;
    state.classList.toggle("error", isError);
}
