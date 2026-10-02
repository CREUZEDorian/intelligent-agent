```mermaid
flowchart TD
    A([User speaks]) --> B[Capture audio<br/>microphone input]
    B --> C{Voice activity<br/>detected?}
    C -- No --> B
    C -- Yes --> D[Speech-to-Text<br/>transcribe audio]
    D --> E[Transcribed text]
    E --> F[POST /chat<br/>FastAPI REST endpoint]
    F --> G[LLM<br/>generate response]
    G --> H[JSON response<br/>returned via REST]
    H --> I[Text-to-Speech<br/>synthesize audio]
    I --> J[Play audio<br/>through speakers]
    J --> K([User hears reply])
    K --> B

    subgraph Client["AI Agent (Client)"]
        B
        C
        D
        E
        I
        J
    end

    subgraph Server["Backend (FastAPI)"]
        F
        G
        H
    end
```
