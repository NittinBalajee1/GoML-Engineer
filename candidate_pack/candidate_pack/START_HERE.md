# Take-home

For the full setup guide, detailed task explanations, validation results, and GitHub-rendered flowcharts, start with [README.md](README.md).

| Part | Folder | Task |
| --- | --- | --- |
| 1 | `placement/` | Predict which registered students get placed |
| 2 | `class_messages/` | Turn short class messages into a task list with a local model |

Spend about a day on each. Start the model download for part 2 before you get deep into part 1.

Part 2 has to run on your own machine. Do not send those messages to a hosted API.

Read the `README.md` in each folder. Submit both parts in one folder:

```
placement/          # analysis.ipynb, predict.py, requirements.txt
class_messages/     # extract.py, NOTES.md, requirements.txt
```

An optional local multi-page web interface for running both workflows is in `frontend/`. Start it from PowerShell with `python server.py` in that folder; see `frontend/README.md`. The tabs navigate to separate pages. Mermaid architecture diagrams are in `frontend/ARCHITECTURE.md`.
