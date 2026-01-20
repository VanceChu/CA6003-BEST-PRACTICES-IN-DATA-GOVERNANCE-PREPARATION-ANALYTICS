Anki Template (MCQ Auto-Flip)

Front Template:
```html
<div class="card">
  {{Front}}
</div>

<script>
(function() {
  function init() {
    if (document.getElementById("answer")) {
      return;
    }
    var mcq = document.querySelector(".mcq");
    if (!mcq) {
      return;
    }
    var answer = mcq.dataset.answer || "";
    var feedback = mcq.querySelector(".feedback");
    var options = mcq.querySelectorAll(".option");
    options.forEach(function(label) {
      var input = label.querySelector("input");
      if (!input) {
        return;
      }
      input.addEventListener("change", function() {
        options.forEach(function(l) {
          l.classList.remove("correct", "wrong");
        });
        if (input.value === answer) {
          label.classList.add("correct");
          if (feedback) {
            feedback.textContent = "Correct";
          }
          if (typeof pycmd !== "undefined") {
            setTimeout(function() {
              pycmd("ans");
            }, 150);
          }
        } else {
          label.classList.add("wrong");
          if (feedback) {
            feedback.textContent = "Wrong, try again";
          }
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
```

Back Template:
```html
{{FrontSide}}

<hr id="answer">

<div class="card">
  {{Back}}
</div>
```

Styling:
```css
.card {
  text-align: left;
}

.mcq .question {
  font-weight: 600;
  margin-bottom: 8px;
}

.mcq .option {
  display: block;
  margin: 6px 0;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 6px;
  cursor: pointer;
}

.mcq .option input {
  margin-right: 8px;
}

.mcq .option.correct {
  border-color: #1b7f2a;
  background: #e6f7e9;
}

.mcq .option.wrong {
  border-color: #b00020;
  background: #fdeaea;
}

.mcq .feedback {
  margin-top: 8px;
  font-weight: 600;
}

.back .answer {
  font-weight: 600;
  margin-bottom: 6px;
}
```
