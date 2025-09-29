# Create Project PRP

Creates a hierarchical, multi-stage project PRP that decomposes complex projects into interconnected PRPs following the Repository Planning Graph (RPG) methodology from arXiv:2509.16198.

## Arguments: $ARGUMENTS

This command takes a high-level project description and creates a master project PRP that:
1. Performs proposal-level planning to identify functional modules and dependencies
2. Creates implementation-level construction plans for each stage
3. Generates a chain of interconnected PRPs that can be executed sequentially
4. Establishes clear data flows and dependencies between project stages

## Core Process

### Stage 1: Proposal-Level Planning

You are creating a multi-stage project plan. First, analyze the project requirements in `$ARGUMENTS` and break it down using hierarchical decomposition:

1. **Project Scope Analysis**: 
   - Extract the core project goals, requirements, and success criteria
   - Identify the primary functional modules needed
   - Determine the target architecture and technology stack

2. **Functional Decomposition**:
   - Break down the project into 3-7 major functional stages/modules
   - Each stage should represent a complete, testable vertical slice
   - Order stages by dependency relationships (data flows, interface requirements)
   - Identify inter-stage dependencies and data contracts

3. **Stage Planning**:
   - For each stage, define:
     - Core functionality and business value
     - Technical requirements and constraints  
     - Input dependencies from previous stages
     - Output deliverables for subsequent stages
     - Success criteria and validation methods

### Stage 2: Implementation-Level Construction

For each identified stage, create implementation planning:

1. **File Structure Planning**:
   - Define folder and file organization for each stage
   - Plan shared components and interfaces between stages
   - Identify reusable patterns and abstractions

2. **Data Flow Design**:
   - Map inter-stage data flows and API contracts
   - Design shared data structures and schemas
   - Plan integration points and handoff mechanisms

3. **Interface Specifications**:
   - Define clear interfaces between stages
   - Specify input/output formats and validation
   - Plan error handling and edge cases

### Stage 3: PRP Chain Generation

Generate the master project PRP following this structure:

```markdown
# Project: [PROJECT_NAME]

## Project Overview
[High-level project description and goals]

## Architecture & Stage Overview
[Visual/textual representation of the stage dependency graph]

## Project Stages

### Stage 1: [STAGE_NAME]
- **Purpose**: [What this stage accomplishes]  
- **Dependencies**: [What it needs from previous stages]
- **Deliverables**: [What it provides to next stages]
- **PRP File**: `PRPs/stage-1-[stage-name].md`

### Stage 2: [STAGE_NAME]
[Repeat for each stage...]

## Execution Plan
1. Execute stages in dependency order
2. Validate each stage before proceeding
3. Integration testing between stages
4. End-to-end system validation

## Stage PRPs
[List of all individual PRP files that will be created]

## Global Context
[Shared patterns, conventions, and constraints that apply across all stages]
```

### Stage 4: Individual Stage PRP Creation

For each stage identified, create a separate PRP file using the standard PRP template but with additional project context:

1. **Enhanced Context Section**: Include references to:
   - Previous stage outputs (if applicable)
   - Next stage requirements (if applicable)  
   - Global project patterns and conventions
   - Inter-stage interface specifications

2. **Dependency Management**: Clearly specify:
   - What data/interfaces this stage consumes
   - What data/interfaces this stage produces
   - Integration points with other stages

3. **Validation Loops**: Include tests for:
   - Stage-specific functionality
   - Integration with previous stages
   - Interface compatibility with next stages

## Implementation Details

Follow these steps to execute this command:

1. **Parse Project Requirements**:
   ```
   Read and analyze the project description from $ARGUMENTS
   Extract key requirements, constraints, and success criteria
   ```

2. **Research Existing Patterns**:
   ```
   Scan the codebase for similar project structures
   Identify existing patterns and conventions to follow
   Check for related architecture documentation
   ```

3. **Generate Project Structure**:
   ```
   Create the master project PRP file: PRPs/project-[name].md
   Generate individual stage PRP files: PRPs/stage-N-[name].md  
   ```

4. **Validate Project Plan**:
   ```
   Check for circular dependencies between stages
   Verify all stages have clear success criteria
   Ensure data flows are well-defined
   ```

## File Naming Convention

- Master Project PRP: `PRPs/project-[project-name].md`
- Stage PRPs: `PRPs/stage-[N]-[stage-name].md` (where N is execution order)
- Project template used: `PRPs/templates/prp_project.md`

## Success Criteria

The command succeeds when:
- [ ] Master project PRP file is created with complete stage breakdown
- [ ] Individual stage PRP files are generated for each identified stage  
- [ ] Dependencies between stages are clearly defined
- [ ] Each stage has clear success criteria and validation methods
- [ ] Integration points between stages are specified
- [ ] The project can be executed by running `prp-project-execute`

## Key Principles from RPG Methodology

- **Hierarchical Planning**: Decompose from high-level goals to concrete implementation
- **Graph-Based Dependencies**: Model stage relationships as a directed acyclic graph
- **Progressive Refinement**: Start with functional requirements, then add implementation details
- **Test-Driven Validation**: Include validation loops at each stage and integration level
- **Context Preservation**: Maintain consistent patterns and conventions across all stages

After creating all files, provide a summary of:
1. Number of stages identified
2. Key dependencies between stages  
3. Critical path for project execution
4. Next steps for project execution

Remember: The goal is to create a comprehensive, executable project plan that can be implemented through a series of interconnected PRPs, each building upon the previous stages in a systematic and validated manner.