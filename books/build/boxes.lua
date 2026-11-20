-- 把 Markdown 里的自定义块转换成 LaTeX 环境
--   ::: {.keypoint}  -> keybox 浅色要点框
--   ::: {.partintro} -> partintro 部引言
--   ::: {.tabcap}    -> \tablecap{编号}{标题}  表题（编号由作者按章编排）
function Div(el)
  if el.classes:includes("keypoint") then
    local out = pandoc.List()
    out:insert(pandoc.RawBlock("latex", "\\begin{keybox}"))
    out:extend(el.content)
    out:insert(pandoc.RawBlock("latex", "\\end{keybox}"))
    return out
  end
  if el.classes:includes("partintro") then
    local out = pandoc.List()
    out:insert(pandoc.RawBlock("latex", "\\begin{partintro}"))
    out:extend(el.content)
    out:insert(pandoc.RawBlock("latex", "\\end{partintro}"))
    return out
  end
  if el.classes:includes("tabcap") then
    local text = pandoc.utils.stringify(el)
    local num, title = text:match("^表%s*([%d%-]+)[%s　]+(.+)$")
    if num then
      return pandoc.RawBlock("latex", "\\tablecap{" .. num .. "}{" .. title .. "}")
    end
    return pandoc.RawBlock("latex", "\\tablecap{}{" .. text .. "}")
  end
end
